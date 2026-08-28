import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base import GlobalScheduler


def _as_np_1d(x, n: int, fill: float = 0.0, dtype=None) -> np.ndarray:
    """Robustly convert x to a length-n 1D numpy array (pads/truncates)."""
    if x is None:
        arr = np.full(n, fill, dtype=dtype if dtype is not None else np.float32)
        return arr
    arr = np.array(x, dtype=dtype) if dtype is not None else np.array(x)
    arr = np.ravel(arr)
    if arr.size == 0:
        return np.full(n, fill, dtype=dtype if dtype is not None else np.float32)
    if arr.size < n:
        pad = np.full(n - arr.size, fill, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.size > n:
        arr = arr[:n]
    return arr


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _bincount_assignments(assignments: np.ndarray, num_dcs: int) -> np.ndarray:
    # assignments are int DC ids in [0, num_dcs)
    return np.bincount(assignments, minlength=num_dcs).astype(np.int32)


def _green_capacity_greedy(desirability, batch_size: int, num_dcs: int,
                           avail_pes=None, green_floor: float = 0.25) -> List[int]:
    """Spread a routing batch across DCs, sending MORE to high-`desirability` (e.g. greener)
    DCs but WITHOUT collapsing the whole batch onto one DC (which overloads it → near-zero
    completion). Each DC gets a target share of the batch
        share[dc] ∝ (desirability_norm[dc] + green_floor) * capacity_weight[dc]
    so the greenest DC receives the largest slice, a `green_floor` keeps brown DCs from being
    starved (guaranteeing the batch spreads), and free-PE capacity (`avail_pes`) gently biases
    away from small DCs. Cloudlets are assigned greedily to whichever DC is furthest below its
    running share target. Returns a list[int] of length batch_size."""
    des = _normalize_01(_as_np_1d(desirability, num_dcs, fill=0.0, dtype=np.float32))
    cap = _as_np_1d(avail_pes, num_dcs, fill=0.0, dtype=np.float64)
    cap = np.maximum(cap, 0.0)
    if cap.sum() <= 0:
        cap = np.ones(num_dcs, dtype=np.float64)
    cap_w = 0.5 + 0.5 * (cap / (cap.max() + 1e-9))     # soft capacity bias in [0.5, 1.0]
    weight = (des + green_floor) * cap_w
    share = weight / weight.sum()                       # fraction of batch per DC
    assigned = np.zeros(num_dcs, dtype=np.float64)
    actions: List[int] = []
    for i in range(batch_size):
        target = share * (i + 1)                        # cumulative target assignments per DC
        deficit = target - assigned                     # most-underserved DC wins the cloudlet
        dc = int(np.argmax(deficit))
        actions.append(dc)
        assigned[dc] += 1.0
    return actions


class RandomGlobalScheduler(GlobalScheduler):
    """Random Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        return np.random.randint(0, self.num_datacenters,
                                 size=self.batch_size).tolist()


class RoundRobinGlobalScheduler(GlobalScheduler):
    """ Round Robin Global Scheduler"""

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        self.current_dc = 0

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        actions = []
        for _ in range(self.batch_size):
            actions.append(self.current_dc)
            self.current_dc = (self.current_dc + 1) % self.num_datacenters
        return actions

    def reset(self):
        self.current_dc = 0


class MinQueueGlobalScheduler(GlobalScheduler):
    """Min-length-WaitingQueue Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        queue_sizes = np.array(global_obs.get('dc_queue_sizes', [0] * self.num_datacenters))
        actions = []
        for _ in range(self.batch_size):
            # Select the DC with the smallest size waiting queue
            best_dc = int(np.argmin(queue_sizes))
            actions.append(best_dc)
            # Simulated Queue Increment (for Continuous Decision-Making)
            queue_sizes[best_dc] += 1
        return actions


class GreenAwareGlobalScheduler(GlobalScheduler):
    """GreenOpt Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        green_ratios = np.array(global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters))
        # Prefer greener DCs but spread across capacity (routing the whole batch to a single
        # greenest DC overloads it → near-zero completion).
        return _green_capacity_greedy(green_ratios, self.batch_size, self.num_datacenters,
                                      global_obs.get('dc_available_pes'))


class GreenForecastAwareGlobalScheduler(GlobalScheduler):
    """SPATIAL-forecast routing: route to the DC with the highest FORECAST green over the
    near future (dc_future_long_mean = the about-to-be-green DC), instead of the
    currently-green DC. Same one-line logic as green_aware but using FUTURE green →
    isolates the spatial-routing value of the forecast (vs green_aware = forecast-blind)."""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        fut = global_obs.get('dc_future_long_mean')
        if fut is None:
            fut = global_obs.get('dc_future_short_mean')
        if fut is None:                       # no forecast available → fall back to green-now
            fut = global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters)
        # Prefer DCs greenest in the near future, spread across capacity (no single-DC collapse).
        return _green_capacity_greedy(fut, self.batch_size, self.num_datacenters,
                                      global_obs.get('dc_available_pes'))


class GreenQueueBalancedGlobalScheduler(GlobalScheduler):
    """Green Energy + Queue Balancing Global Scheduling: Comprehensive consideration of green energy proportion and queue length"""

    def __init__(self, num_datacenters: int, batch_size: int, green_weight: float = 0.6):
        super().__init__(num_datacenters, batch_size)
        self.green_weight = green_weight

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        green_ratios = np.array(global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters))
        queue_sizes = np.array(global_obs.get('dc_queue_sizes', [0] * self.num_datacenters))

        # Normalization
        green_norm = green_ratios / (green_ratios.max() + 1e-8)
        queue_norm = 1 - (queue_sizes / (queue_sizes.max() + 1e-8))

        # Conprehensive Scoring
        scores = self.green_weight * green_norm + (1 - self.green_weight) * queue_norm

        actions = []
        temp_queues = queue_sizes.copy()
        for _ in range(self.batch_size):
            # Dynamic Update Score
            queue_norm = 1 - (temp_queues / (temp_queues.max() + 1e-8))
            scores = self.green_weight * green_norm + (1 - self.green_weight) * queue_norm
            best_dc = int(np.argmax(scores))
            actions.append(best_dc)
            temp_queues[best_dc] += 1

        return actions


class GreenForecastQueueBalancedGlobalScheduler(GreenQueueBalancedGlobalScheduler):
    """T3 重做(Codex 批准 2026-08-27):**容量感知的全知臂**。

    与 `GreenQueueBalancedGlobalScheduler` **逐行相同**,只把绿电信号从
    `dc_green_ratio`(当前)换成 `dc_future_long_mean`(未来)。因此
    (green_queue_balanced, green_forecast_queue_balanced) 构成**匹配对**,
    唯一差异 = 当前绿电 vs 未来绿电信号。

    为什么需要它:T3 首轮的全知臂 `green_forecast` 走 `_green_capacity_greedy`,
    那里的容量只是**每批算一次的软偏置**(cap_w∈[0.5,1.0],至多 2× 调制),
    没有批内拥塞反馈 ⇒ 未来最绿的 DC 仍被压垮(完成率 67%、绿电用 350 Wh /
    弃 7035 Wh),全知臂反而输给盲态,S_ach 只能报 undefined。
    本类改用 green_queue_balanced 已验证的**批内动态队列反馈**。
    """

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        fut = global_obs.get('dc_future_long_mean')
        if fut is None:
            fut = global_obs.get('dc_future_short_mean')
        if fut is None:                       # 无预报 -> 退回当前绿电
            fut = global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters)
        # 2026-08-28 实测:Java 在**无绿电序列**的 DC 上返回"无数据"默认值
        # features[2]=0.5(GreenEnergyProvider.computeFutureTrendFeatures),
        # 而真实读数均值只有 0.242 ⇒ 无绿电且最脏的 DC3(0.75)/DC4(0.92)
        # 反而成了 argmax 的首选。盲态臂用 dc_green_ratio(无绿电 DC 正确为 0)
        # 不受影响。这里按"是否具备绿电能力"遮罩,使两臂回到同一基础上。
        fut = np.asarray(fut, dtype=np.float64).reshape(-1)[:self.num_datacenters]
        cap = np.asarray(global_obs.get('dc_current_green_power_w',
                                        [0.0] * self.num_datacenters),
                         dtype=np.float64).reshape(-1)[:self.num_datacenters]
        seen = getattr(self, '_green_capable', None)
        if seen is None or len(seen) != self.num_datacenters:
            seen = np.zeros(self.num_datacenters, dtype=bool)
        seen = seen | (cap > 0.0)
        self._green_capable = seen
        if seen.any():
            fut = np.where(seen, fut, 0.0)
        shim = dict(global_obs)
        shim['dc_green_ratio'] = fut          # 唯一改动:信号换成(遮罩后的)未来绿电
        return super().schedule(shim)

    def reset(self):
        super().reset()
        self._green_capable = None


class MinBrownPowerGlobalScheduler(GlobalScheduler):
    """
    Min-Brown-Power Global Scheduler:

    Choose the datacenter with the minimum estimated brown power usage:
      brown_power ~= dc_current_power_w * (1 - dc_green_ratio)

    Falls back to maximizing green ratio if power signals are missing.
    """

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), self.num_datacenters, fill=0.5, dtype=np.float32)
        current_power = _as_np_1d(global_obs.get("dc_current_power_w"), self.num_datacenters, fill=np.nan, dtype=np.float32)

        if np.isfinite(current_power).any():
            # Replace non-finite with max finite to avoid selecting missing signals
            finite = current_power[np.isfinite(current_power)]
            max_finite = float(np.max(finite)) if finite.size else 0.0
            cur = np.where(np.isfinite(current_power), current_power, max_finite)
            brown = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
            desirability = -brown              # less brown power == more desirable
        else:
            desirability = green_ratio

        # Prefer least-brown DCs, spread across capacity (no single-DC collapse).
        return _green_capacity_greedy(desirability, self.batch_size, self.num_datacenters,
                                      global_obs.get('dc_available_pes'))


class WeightedScoreGlobalScheduler(GlobalScheduler):
    """
    Weighted multi-criteria heuristic (no learning).

    Score per DC (higher is better):
      + w_green * norm(green_ratio)
      + w_avail * norm(available_pes)
      + w_idle_green * norm(current_green_power_w)   (optional)
      - w_queue * norm(queue_size)
      - w_util  * norm(utilization)
      - w_brown * norm(brown_power)                  (optional)
    """

    def __init__(
        self,
        num_datacenters: int,
        batch_size: int,
        w_green: float = 0.45,
        w_queue: float = 0.25,
        w_util: float = 0.15,
        w_avail: float = 0.10,
        w_brown: float = 0.05,
        w_idle_green: float = 0.0,
    ):
        super().__init__(num_datacenters, batch_size)
        self.w_green = w_green
        self.w_queue = w_queue
        self.w_util = w_util
        self.w_avail = w_avail
        self.w_brown = w_brown
        self.w_idle_green = w_idle_green

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), n, fill=0.5, dtype=np.float32)
        queue = _as_np_1d(global_obs.get("dc_queue_sizes"), n, fill=0.0, dtype=np.float32)
        util = _as_np_1d(global_obs.get("dc_utilizations"), n, fill=0.0, dtype=np.float32)
        avail = _as_np_1d(global_obs.get("dc_available_pes"), n, fill=0.0, dtype=np.float32)
        current_green = _as_np_1d(global_obs.get("dc_current_green_power_w"), n, fill=0.0, dtype=np.float32)
        current_power = _as_np_1d(global_obs.get("dc_current_power_w"), n, fill=np.nan, dtype=np.float32)

        # Optional brown power signal
        if np.isfinite(current_power).any():
            finite = current_power[np.isfinite(current_power)]
            max_finite = float(np.max(finite)) if finite.size else 0.0
            cur = np.where(np.isfinite(current_power), current_power, max_finite)
            brown = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
        else:
            brown = np.zeros(n, dtype=np.float32)

        score = (
            self.w_green * _normalize_01(green_ratio)
            + self.w_avail * _normalize_01(avail)
            + self.w_idle_green * _normalize_01(current_green)
            - self.w_queue * _normalize_01(queue)
            - self.w_util * _normalize_01(util)
            - self.w_brown * _normalize_01(brown)
        )

        # Greedy per cloudlet with queue update to reduce hotspotting
        actions: List[int] = []
        tmp_queue = queue.copy()
        green_norm = _normalize_01(green_ratio)
        util_norm = _normalize_01(util)
        avail_norm = _normalize_01(avail)
        brown_norm = _normalize_01(brown)
        idle_green_norm = _normalize_01(current_green)

        for _ in range(self.batch_size):
            queue_norm = _normalize_01(tmp_queue)
            tmp_score = (
                self.w_green * green_norm
                + self.w_avail * avail_norm
                + self.w_idle_green * idle_green_norm
                - self.w_queue * queue_norm
                - self.w_util * util_norm
                - self.w_brown * brown_norm
            )
            best = int(np.argmax(tmp_score))
            actions.append(best)
            tmp_queue[best] += 1.0

        return actions


class GeneticAlgorithmGlobalScheduler(GlobalScheduler):
    """
    GA baseline for batch routing.

    Chromosome: length=batch_size, each gene is a DC id.
    Fitness (lower is better): queue pressure + imbalance + brown power - green preference + capacity violation penalty.

    Notes:
    - Designed to be lightweight (small pop/iters) for per-step online scheduling.
    - Uses only signals present in baseline evaluate converter; extra keys are optional.
    - IMPROVED: Uses dc_current_green_power_w as fallback when green_ratio is unavailable.
    - IMPROVED: Better capacity-aware load balancing across DCs.
    """

    def __init__(
        self,
        num_datacenters: int,
        batch_size: int,
        population_size: int = 30,
        generations: int = 15,
        mutation_rate: float = 0.10,
        tournament_k: int = 3,
        w_queue: float = 1.0,
        w_imbalance: float = 0.8,
        w_util: float = 0.2,
        w_brown: float = 0.3,
        w_green: float = 0.5,
        w_green_power: float = 0.4,
        w_capacity: float = 1.5,
        w_capacity_balance: float = 0.6,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(num_datacenters, batch_size)
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = max(2, int(tournament_k))
        self.w_queue = float(w_queue)
        self.w_imbalance = float(w_imbalance)
        self.w_util = float(w_util)
        self.w_brown = float(w_brown)
        self.w_green = float(w_green)
        self.w_green_power = float(w_green_power)
        self.w_capacity = float(w_capacity)
        self.w_capacity_balance = float(w_capacity_balance)
        self.rng = np.random.default_rng(rng_seed)

    def _fitness(
        self,
        assignment: np.ndarray,
        queue: np.ndarray,
        green_ratio: np.ndarray,
        green_power: np.ndarray,
        brown_power: np.ndarray,
        utilizations: np.ndarray,
        avail_pes: np.ndarray,
        batch_pes: np.ndarray,
    ) -> float:
        n = self.num_datacenters
        add = _bincount_assignments(assignment, n).astype(np.float32)
        q_pred = queue.astype(np.float32) + add
        queue_pressure = float(np.mean(q_pred))
        imbalance = float(np.std(q_pred))

        # Prefer higher green_ratio on chosen DCs
        green_term = float(np.mean(green_ratio[assignment])) if green_ratio.size else 0.0

        # NEW: Prefer higher green_power (current green energy availability) on chosen DCs
        green_power_term = float(np.mean(green_power[assignment])) if green_power.size else 0.0
        # Normalize green_power_term to [0, 1] range for consistent weighting
        if green_power.max() > 0:
            green_power_term = green_power_term / green_power.max()

        brown_term = float(np.mean(brown_power[assignment])) if brown_power.size else 0.0
        # Normalize brown_term
        if brown_power.max() > 0:
            brown_term = brown_term / brown_power.max()

        # Utilization penalty (soft): avoid sending more to already-busy DCs.
        util_pen = float(np.mean(utilizations[assignment])) if utilizations.size else 0.0

        # Capacity penalty (DC-aggregated): penalize concentrating demand onto DCs
        # whose available PEs are insufficient.
        cap_pen = 0.0
        if avail_pes.size and batch_pes.size:
            demands = batch_pes[: assignment.size].astype(np.float32)
            demand_sum = np.zeros(n, dtype=np.float32)
            # accumulate demand per DC
            np.add.at(demand_sum, assignment, demands)
            shortfall_dc = np.maximum(0.0, demand_sum - avail_pes.astype(np.float32))
            # normalize by batch size to keep scale stable across batch sizes
            cap_pen = float(np.sum(shortfall_dc) / max(1, assignment.size))

        # NEW: Capacity balance penalty - prefer distributing load proportional to DC capacity
        cap_balance_pen = 0.0
        if avail_pes.size and avail_pes.sum() > 0:
            # Ideal distribution: proportional to available PEs
            total_avail = avail_pes.sum()
            ideal_ratio = avail_pes / total_avail
            actual_ratio = add / max(1, add.sum())
            # Penalize deviation from ideal distribution
            cap_balance_pen = float(np.sum(np.abs(actual_ratio - ideal_ratio)))

        return (
            self.w_queue * queue_pressure
            + self.w_imbalance * imbalance
            + self.w_util * util_pen
            + self.w_brown * brown_term
            - self.w_green * green_term
            - self.w_green_power * green_power_term
            + self.w_capacity * cap_pen
            + self.w_capacity_balance * cap_balance_pen
        )

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idxs = self.rng.integers(0, fitnesses.size, size=self.tournament_k)
        best = idxs[np.argmin(fitnesses[idxs])]
        return int(best)

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        b = self.batch_size

        queue = _as_np_1d(global_obs.get("dc_queue_sizes"), n, fill=0.0, dtype=np.float32)
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), n, fill=0.0, dtype=np.float32)
        utilizations = _as_np_1d(global_obs.get("dc_utilizations"), n, fill=0.0, dtype=np.float32)
        avail_pes = _as_np_1d(global_obs.get("dc_available_pes"), n, fill=1.0, dtype=np.float32)
        batch_pes = np.ravel(np.array(global_obs.get("batch_cloudlet_pes", []), dtype=np.float32))
        if batch_pes.size < b:
            batch_pes = np.ones(b, dtype=np.float32)

        # Get current green power (available green energy at each DC)
        green_power = _as_np_1d(global_obs.get("dc_current_green_power_w"), n, fill=0.0, dtype=np.float32)

        current_power = _as_np_1d(global_obs.get("dc_current_power_w"), n, fill=np.nan, dtype=np.float32)
        if np.isfinite(current_power).any():
            finite = current_power[np.isfinite(current_power)]
            max_finite = float(np.max(finite)) if finite.size else 0.0
            cur = np.where(np.isfinite(current_power), current_power, max_finite)
            brown_power = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
        else:
            brown_power = np.zeros(n, dtype=np.float32)

        # --- init population (mix of heuristics + random) ---
        pop = np.empty((self.population_size, b), dtype=np.int32)
        seed_idx = 0

        # Seed 0: greedy min-queue (with simulated increments)
        tmp_q = queue.copy()
        seed0 = np.empty(b, dtype=np.int32)
        for i in range(b):
            dc = int(np.argmin(tmp_q))
            seed0[i] = dc
            tmp_q[dc] += 1.0
        pop[seed_idx] = seed0
        seed_idx += 1

        # Seed 1: greedy weighted score (green_ratio vs queue)
        if seed_idx < self.population_size:
            tmp_q = queue.copy()
            seed1 = np.empty(b, dtype=np.int32)
            g_norm = _normalize_01(green_ratio)
            for i in range(b):
                q_norm = _normalize_01(tmp_q)
                score = 0.6 * g_norm + 0.4 * (1.0 - q_norm)
                dc = int(np.argmax(score))
                seed1[i] = dc
                tmp_q[dc] += 1.0
            pop[seed_idx] = seed1
            seed_idx += 1

        # Seed 2: NEW - greedy green_power (prefer DCs with high current green power)
        if seed_idx < self.population_size:
            tmp_q = queue.copy()
            seed2 = np.empty(b, dtype=np.int32)
            gp_norm = _normalize_01(green_power)
            for i in range(b):
                q_norm = _normalize_01(tmp_q)
                # Combine green power preference with queue avoidance
                score = 0.5 * gp_norm + 0.3 * (1.0 - q_norm) + 0.2 * _normalize_01(avail_pes)
                dc = int(np.argmax(score))
                seed2[i] = dc
                tmp_q[dc] += 1.0
            pop[seed_idx] = seed2
            seed_idx += 1

        # Seed 3: NEW - capacity-proportional distribution (load balance by DC capacity)
        if seed_idx < self.population_size:
            seed3 = np.empty(b, dtype=np.int32)
            total_avail = avail_pes.sum()
            if total_avail > 0:
                # Distribute cloudlets proportionally to available PEs
                probs = avail_pes / total_avail
                seed3 = self.rng.choice(n, size=b, p=probs).astype(np.int32)
            else:
                seed3 = self.rng.integers(0, n, size=b, dtype=np.int32)
            pop[seed_idx] = seed3
            seed_idx += 1

        # Seed 4: NEW - round-robin across all DCs (ensures all DCs get some load)
        if seed_idx < self.population_size:
            seed4 = np.array([i % n for i in range(b)], dtype=np.int32)
            pop[seed_idx] = seed4
            seed_idx += 1

        # Seed 5: NEW - combined green_power + capacity score
        if seed_idx < self.population_size:
            tmp_q = queue.copy()
            seed5 = np.empty(b, dtype=np.int32)
            gp_norm = _normalize_01(green_power)
            cap_norm = _normalize_01(avail_pes)
            for i in range(b):
                q_norm = _normalize_01(tmp_q)
                # Heavy emphasis on green power and capacity
                score = 0.4 * gp_norm + 0.4 * cap_norm + 0.2 * (1.0 - q_norm)
                dc = int(np.argmax(score))
                seed5[i] = dc
                tmp_q[dc] += 1.0
            pop[seed_idx] = seed5
            seed_idx += 1

        # Remaining: random
        for i in range(seed_idx, self.population_size):
            pop[i] = self.rng.integers(0, n, size=b, dtype=np.int32)

        fitnesses = np.array(
            [self._fitness(ind, queue, green_ratio, green_power, brown_power, utilizations, avail_pes, batch_pes) for ind in pop],
            dtype=np.float32,
        )

        # --- evolve ---
        for _ in range(self.generations):
            new_pop = np.empty_like(pop)
            # Elitism: keep best
            elite_idx = int(np.argmin(fitnesses))
            new_pop[0] = pop[elite_idx]

            for i in range(1, self.population_size):
                p1 = pop[self._tournament(fitnesses)]
                p2 = pop[self._tournament(fitnesses)]

                # Uniform crossover
                mask = self.rng.random(b) < 0.5
                child = np.where(mask, p1, p2).astype(np.int32)

                # Mutation
                mut = self.rng.random(b) < self.mutation_rate
                if mut.any():
                    child[mut] = self.rng.integers(0, n, size=int(mut.sum()), dtype=np.int32)

                new_pop[i] = child

            pop = new_pop
            fitnesses = np.array(
                [self._fitness(ind, queue, green_ratio, green_power, brown_power, utilizations, avail_pes, batch_pes) for ind in pop],
                dtype=np.float32,
            )

        best_idx = int(np.argmin(fitnesses))
        return pop[best_idx].tolist()


class ParticleSwarmGlobalScheduler(GlobalScheduler):
    """
    Lightweight PSO baseline for batch routing (discrete via rounding).

    Particle position: real-valued vector (len=batch_size) mapped to DC ids by rounding+clamp.
    Objective: same as GA (queue pressure + imbalance + brown power - green + capacity penalty).

    IMPROVED: Uses dc_current_green_power_w as fallback when green_ratio is unavailable.
    IMPROVED: Better capacity-aware load balancing across DCs.
    """

    def __init__(
        self,
        num_datacenters: int,
        batch_size: int,
        swarm_size: int = 25,
        iterations: int = 15,
        inertia: float = 0.6,
        c1: float = 1.5,
        c2: float = 1.5,
        w_queue: float = 1.0,
        w_imbalance: float = 0.8,
        w_util: float = 0.2,
        w_brown: float = 0.3,
        w_green: float = 0.5,
        w_green_power: float = 0.4,
        w_capacity: float = 1.5,
        w_capacity_balance: float = 0.6,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(num_datacenters, batch_size)
        self.swarm_size = max(4, int(swarm_size))
        self.iterations = max(1, int(iterations))
        self.inertia = float(inertia)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_queue = float(w_queue)
        self.w_imbalance = float(w_imbalance)
        self.w_util = float(w_util)
        self.w_brown = float(w_brown)
        self.w_green = float(w_green)
        self.w_green_power = float(w_green_power)
        self.w_capacity = float(w_capacity)
        self.w_capacity_balance = float(w_capacity_balance)
        self.rng = np.random.default_rng(rng_seed)

    def _fitness(
        self,
        assignment: np.ndarray,
        queue: np.ndarray,
        green_ratio: np.ndarray,
        green_power: np.ndarray,
        brown_power: np.ndarray,
        utilizations: np.ndarray,
        avail_pes: np.ndarray,
        batch_pes: np.ndarray,
    ) -> float:
        n = self.num_datacenters
        add = _bincount_assignments(assignment, n).astype(np.float32)
        q_pred = queue.astype(np.float32) + add
        queue_pressure = float(np.mean(q_pred))
        imbalance = float(np.std(q_pred))

        green_term = float(np.mean(green_ratio[assignment])) if green_ratio.size else 0.0

        # NEW: Prefer higher green_power (current green energy availability)
        green_power_term = float(np.mean(green_power[assignment])) if green_power.size else 0.0
        if green_power.max() > 0:
            green_power_term = green_power_term / green_power.max()

        brown_term = float(np.mean(brown_power[assignment])) if brown_power.size else 0.0
        if brown_power.max() > 0:
            brown_term = brown_term / brown_power.max()

        util_pen = float(np.mean(utilizations[assignment])) if utilizations.size else 0.0

        cap_pen = 0.0
        if avail_pes.size and batch_pes.size:
            demands = batch_pes[: assignment.size].astype(np.float32)
            demand_sum = np.zeros(n, dtype=np.float32)
            np.add.at(demand_sum, assignment, demands)
            shortfall_dc = np.maximum(0.0, demand_sum - avail_pes.astype(np.float32))
            cap_pen = float(np.sum(shortfall_dc) / max(1, assignment.size))

        # NEW: Capacity balance penalty
        cap_balance_pen = 0.0
        if avail_pes.size and avail_pes.sum() > 0:
            total_avail = avail_pes.sum()
            ideal_ratio = avail_pes / total_avail
            actual_ratio = add / max(1, add.sum())
            cap_balance_pen = float(np.sum(np.abs(actual_ratio - ideal_ratio)))

        return (
            self.w_queue * queue_pressure
            + self.w_imbalance * imbalance
            + self.w_util * util_pen
            + self.w_brown * brown_term
            - self.w_green * green_term
            - self.w_green_power * green_power_term
            + self.w_capacity * cap_pen
            + self.w_capacity_balance * cap_balance_pen
        )

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        b = self.batch_size

        queue = _as_np_1d(global_obs.get("dc_queue_sizes"), n, fill=0.0, dtype=np.float32)
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), n, fill=0.0, dtype=np.float32)
        utilizations = _as_np_1d(global_obs.get("dc_utilizations"), n, fill=0.0, dtype=np.float32)
        avail_pes = _as_np_1d(global_obs.get("dc_available_pes"), n, fill=1.0, dtype=np.float32)
        batch_pes = np.ravel(np.array(global_obs.get("batch_cloudlet_pes", []), dtype=np.float32))
        if batch_pes.size < b:
            batch_pes = np.ones(b, dtype=np.float32)

        # Get current green power
        green_power = _as_np_1d(global_obs.get("dc_current_green_power_w"), n, fill=0.0, dtype=np.float32)

        current_power = _as_np_1d(global_obs.get("dc_current_power_w"), n, fill=np.nan, dtype=np.float32)
        if np.isfinite(current_power).any():
            finite = current_power[np.isfinite(current_power)]
            max_finite = float(np.max(finite)) if finite.size else 0.0
            cur = np.where(np.isfinite(current_power), current_power, max_finite)
            brown_power = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
        else:
            brown_power = np.zeros(n, dtype=np.float32)

        # init swarm with diverse seed particles
        pos = self.rng.random((self.swarm_size, b), dtype=np.float32) * float(max(n - 1, 1))
        vel = (self.rng.random((self.swarm_size, b), dtype=np.float32) - 0.5) * 0.5
        seed_idx = 0

        # Seed 0: greedy min-queue
        if seed_idx < self.swarm_size:
            tmp_q = queue.copy()
            seed0 = np.empty(b, dtype=np.float32)
            for i in range(b):
                dc = int(np.argmin(tmp_q))
                seed0[i] = float(dc)
                tmp_q[dc] += 1.0
            pos[seed_idx] = seed0
            seed_idx += 1

        # Seed 1: greedy green_power + capacity
        if seed_idx < self.swarm_size:
            tmp_q = queue.copy()
            seed1 = np.empty(b, dtype=np.float32)
            gp_norm = _normalize_01(green_power)
            cap_norm = _normalize_01(avail_pes)
            for i in range(b):
                q_norm = _normalize_01(tmp_q)
                score = 0.4 * gp_norm + 0.4 * cap_norm + 0.2 * (1.0 - q_norm)
                dc = int(np.argmax(score))
                seed1[i] = float(dc)
                tmp_q[dc] += 1.0
            pos[seed_idx] = seed1
            seed_idx += 1

        # Seed 2: capacity-proportional distribution
        if seed_idx < self.swarm_size:
            total_avail = avail_pes.sum()
            if total_avail > 0:
                probs = avail_pes / total_avail
                seed2 = self.rng.choice(n, size=b, p=probs).astype(np.float32)
            else:
                seed2 = self.rng.random(b, dtype=np.float32) * float(max(n - 1, 1))
            pos[seed_idx] = seed2
            seed_idx += 1

        # Seed 3: greedy green_power only
        if seed_idx < self.swarm_size:
            tmp_q = queue.copy()
            seed3 = np.empty(b, dtype=np.float32)
            gp_norm = _normalize_01(green_power)
            for i in range(b):
                q_norm = _normalize_01(tmp_q)
                score = 0.7 * gp_norm + 0.3 * (1.0 - q_norm)
                dc = int(np.argmax(score))
                seed3[i] = float(dc)
                tmp_q[dc] += 1.0
            pos[seed_idx] = seed3
            seed_idx += 1

        def to_assign(p: np.ndarray) -> np.ndarray:
            a = np.rint(p).astype(np.int32)
            return np.clip(a, 0, n - 1)

        pbest_pos = pos.copy()
        pbest_fit = np.array(
            [self._fitness(to_assign(p), queue, green_ratio, green_power, brown_power, utilizations, avail_pes, batch_pes) for p in pos],
            dtype=np.float32,
        )
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        for _ in range(self.iterations):
            r1 = self.rng.random((self.swarm_size, b), dtype=np.float32)
            r2 = self.rng.random((self.swarm_size, b), dtype=np.float32)
            vel = (
                self.inertia * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos[None, :] - pos)
            )
            pos = pos + vel
            pos = np.clip(pos, 0.0, float(max(n - 1, 1)))

            fits = np.array(
                [self._fitness(to_assign(p), queue, green_ratio, green_power, brown_power, utilizations, avail_pes, batch_pes) for p in pos],
                dtype=np.float32,
            )
            improved = fits < pbest_fit
            if improved.any():
                pbest_fit = np.where(improved, fits, pbest_fit)
                pbest_pos[improved] = pos[improved]
                new_gbest_idx = int(np.argmin(pbest_fit))
                new_gbest_fit = float(pbest_fit[new_gbest_idx])
                if new_gbest_fit < gbest_fit:
                    gbest_fit = new_gbest_fit
                    gbest_pos = pbest_pos[new_gbest_idx].copy()

        return to_assign(gbest_pos).tolist()


class RLlibGlobalScheduler(GlobalScheduler):
    """
    RLlib-based Global Scheduler (for Multi-DC training with Ray)

    Uses a pre-loaded RLlib Algorithm for inference.
    """

    def __init__(self, num_datacenters: int, batch_size: int, algo):
        """
        Args:
            num_datacenters: number of datacenters
            batch_size: number of cloudlets per step
            algo: loaded RLlib Algorithm instance
        """
        super().__init__(num_datacenters, batch_size)
        self.algo = algo
        self.policy_id = "global_policy"

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        """
        use RLlib model to select target DC for cloudlets in batch
        """
        # Wrap observation to match training-time PettingZoo format:
        # {"observation": <global_obs_dict>}
        wrapped_obs = {
            "observation": global_obs
        }

        action = self.algo.compute_single_action(
            wrapped_obs,
            policy_id=self.policy_id,
            explore=False
        )

        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, (list, tuple)):
            return list(action)
        else:
            return [int(action)] * self.batch_size


def load_rllib_algorithm(checkpoint_path: str, py4j_port_override: int | None = None):
    """
    load RLlib checkpoint and return Algorithm instance

    Args:
        checkpoint_path: checkpoint path
        py4j_port_override: if provided, override env_config['py4j_port'] when creating envs during
            checkpoint restore (prevents colliding with a running training gateway).

    Returns:
        RLlib Algorithm instance
    """
    import ray
    from ray import tune
    from pathlib import Path
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # register custom models
    from src.models.masked_action_model import MaskedActionModel, DictObsModel
    try:
        ModelCatalog.register_custom_model('masked_action_model', MaskedActionModel)
        ModelCatalog.register_custom_model('dict_obs_model', DictObsModel)
    except Exception:
        pass

    # Ensure RLModule classes are importable before restoring New API Stack checkpoints.
    # (Algorithm.from_checkpoint may need to import the module path that defines the RLModule.)
    try:
        import src.models.rlmodule_models  # noqa: F401
        import src.models.rlmodule_resmlp_models  # noqa: F401
        import src.models.rlmodule_gmlp_models  # noqa: F401
    except Exception:
        # If these modules are not present/used, it's fine.
        pass

    # register environment
    from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv

    def env_creator(cfg):
        # IMPORTANT: RLlib may create envs during Algorithm.from_checkpoint()
        # (space inference / env runner init). The cfg restored from the
        # checkpoint typically has a baked-in py4j_port from training time
        # (often 25333). Always override:
        #   * explicit override -> use it
        #   * None              -> strip the port so the env auto-launches
        #                          its own gateway via _find_free_port +
        #                          gradlew run subprocess
        # Also ensure gateway_log_dir exists; the auto-launch path requires it
        # but the checkpoint cfg may have been pickled without it.
        if isinstance(cfg, dict):
            cfg = dict(cfg)
            if py4j_port_override is not None:
                cfg["py4j_port"] = int(py4j_port_override)
            else:
                cfg["py4j_port"] = None
            cfg.setdefault(
                "gateway_log_dir",
                str(Path(checkpoint_path).resolve().parent / "eval_gateways"),
            )
        env = HierarchicalMultiDCParallelEnv(cfg)
        return ParallelPettingZooEnv(env)

    tune.register_env('multidc_env', env_creator)

    # load checkpoint (using absolute file:// URI, to avoid pyarrow error for relative paths)
    checkpoint_uri = Path(checkpoint_path).resolve().as_uri()
    algo = Algorithm.from_checkpoint(checkpoint_uri)
    return algo


class RLlibNewAPIGlobalScheduler(GlobalScheduler):
    """
    RLlib-based Global Scheduler for New API Stack (RLModule).

    Uses RLModule directly for inference instead of compute_single_action.
    This is required for models trained with enable_rl_module_and_learner=True.
    """

    def __init__(self, num_datacenters: int, batch_size: int, algo, stochastic: bool = False):
        """
        Args:
            num_datacenters: number of datacenters
            batch_size: number of cloudlets per step
            algo: loaded RLlib Algorithm instance (New API Stack)
            stochastic: if True, sample each per-slot routing choice from the policy's
                categorical distribution instead of taking argmax. Needed for a faithful
                iso-completion comparison: with 128 simultaneous routing slots seeing
                identical obs, greedy argmax collapses all slots onto a single DC (overload),
                whereas the trained policy relies on sampling to spread load across DCs.
        """
        super().__init__(num_datacenters, batch_size)
        self.algo = algo
        self.policy_id = "global_policy"
        self.stochastic = stochastic

        # Get the RLModule from the algorithm
        self._rl_module = self._get_rl_module()

        # EU-CRD deployment-time trust sentinel (env-gated via TRUST_GATE_MODE;
        # see src/baselines/trust_sentinel.py). Loads the trained Q-ensemble
        # from the learner-side checkpoint files — the env-runner module may be
        # inference-only with randomly-initialised q_heads.
        self._sentinel = None
        import os as _os
        if _os.environ.get("TRUST_GATE_MODE", "").strip():
            source = _os.environ.get("TRUST_GATE_SOURCE", "qvar").strip().lower()
            if source == "resid":
                # Forecast-verification monitor: audits forecast vs realized
                # green online; no Q-ensemble needed (vanilla ckpts work too).
                from src.baselines.trust_sentinel import ForecastResidualMonitor
                self._sentinel = ForecastResidualMonitor.from_env(num_slots=batch_size)
            else:
                from src.baselines.trust_sentinel import TrustSentinel
                ckpt = _os.environ.get("EVAL_CHECKPOINT_PATH", "").strip()
                if not ckpt:
                    raise RuntimeError(
                        "TRUST_GATE_MODE is set but EVAL_CHECKPOINT_PATH is empty — "
                        "run via evaluate.py --checkpoint so the sentinel can load "
                        "the Q-ensemble weights."
                    )
                self._sentinel = TrustSentinel.from_env(
                    ckpt, num_slots=batch_size, policy_id=self.policy_id
                )
            print(
                f"[TrustSentinel] active: {self._sentinel.summary()} "
                f"thresh={self._sentinel.threshold}"
            )

    def _get_rl_module(self):
        """Get the RLModule for inference."""
        try:
            env_runner = getattr(self.algo, 'env_runner', None)
            if env_runner is not None:
                module = getattr(env_runner, 'module', None)
                if module is not None:
                    return module
        except Exception as e:
            print(f"Warning: Could not get RLModule: {e}")
        return None

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        """
        Use RLModule to select target DC for cloudlets in batch.
        """
        import torch

        if self._rl_module is None:
            raise RuntimeError("RLModule not available. Make sure algorithm uses New API Stack.")

        # Trust sentinel (resid source): audit the RAW obs (forecast vs realized
        # green) BEFORE the policy sees them, then — in repair mode — hand the
        # policy a repaired view (lying DCs' forecast features inverted). The
        # ring buffer keeps the raw stream, so the trust estimate can't oscillate.
        if self._sentinel is not None and hasattr(self._sentinel, "measure_obs"):
            self._sentinel.measure_obs(global_obs)
            if hasattr(self._sentinel, "repair"):
                global_obs = self._sentinel.repair(global_obs)

        # Wrap observation to match training-time PettingZoo format
        wrapped_obs = {"observation": global_obs}

        # Get the global policy module
        module = self._rl_module
        if hasattr(module, '__getitem__'):
            module = module[self.policy_id]

        # Prepare batch input
        batch = self._obs_to_batch(wrapped_obs)

        # Forward pass
        with torch.no_grad():
            output = module.forward_inference(batch)

        # Trust sentinel (qvar source): epistemic disagreement on the trunk
        # features the policy just used (ensemble forward hook). Gating for
        # both sources happens in _sample_multidiscrete.
        if self._sentinel is not None and not hasattr(self._sentinel, "measure_obs"):
            self._sentinel.measure(getattr(module, "_captured_features", None))

        # Extract action
        if "actions" in output:
            if self._sentinel is not None:
                raise RuntimeError(
                    "TrustSentinel needs action_dist_inputs to gate/log per-"
                    "decision, but the module returned pre-sampled 'actions'."
                )
            actions = output["actions"]
        elif "action_dist_inputs" in output:
            # For MultiDiscrete, action_dist_inputs shape: (batch, sum of nvec).
            # Recurrent backbones (e.g. GTrXL) emit [B, T, sum-of-nvec]; squeeze
            # the singleton T so downstream reshape works.
            dist_inputs = output["action_dist_inputs"]
            if dist_inputs.dim() == 3 and dist_inputs.shape[1] == 1:
                dist_inputs = dist_inputs.squeeze(1)
            actions = self._sample_multidiscrete(dist_inputs)
        else:
            raise ValueError(f"Unknown output format: {output.keys()}")

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        actions = actions.flatten()
        return actions.tolist()

    def _sample_multidiscrete(self, dist_inputs) -> np.ndarray:
        """Sample from MultiDiscrete distribution inputs."""
        import torch

        # dist_inputs shape: (batch, batch_size * num_choices). The per-slot choice
        # count is num_datacenters (+1 when the global DEFER head is active), so
        # derive it from the actual tensor size instead of hardcoding num_datacenters
        # — otherwise an arch-B defer checkpoint (6 logits/slot) mis-reshapes against 5.
        batch_size = dist_inputs.shape[0]
        num_choices = dist_inputs.shape[-1] // self.batch_size
        logits = dist_inputs.reshape(batch_size, self.batch_size, num_choices)

        # Trust sentinel: in gate mode, suppress the DEFER column (last choice)
        # when the just-measured ensemble disagreement exceeds the threshold —
        # the policy then falls back to reactive run-now with its learned
        # spatial routing intact. In log mode this only records the signal.
        if self._sentinel is not None:
            logits = self._sentinel.maybe_gate(logits)

        # DECODE_TOPK (env-gated experiment): deterministic PROPORTIONAL-ALLOCATION decode for an
        # allocation policy. Per-slot argmax wrongly collapses a homogeneous batch onto one DC; this
        # instead spreads the batch ∝ the policy's mean distribution, but only over its TOP-K choices
        # (drops the low-prob — typically brown — DCs). Deterministic + reproducible (no sampling).
        import os as _os
        _topk = int(_os.environ.get("DECODE_TOPK", "0") or 0)
        if _topk > 0:
            probs = torch.softmax(logits, dim=-1)                  # (B, n_slots, n_choices)
            pbar = probs.mean(dim=1)                               # (B, n_choices) batch alloc dist
            k = min(_topk, pbar.shape[-1])
            kth = torch.topk(pbar, k, dim=-1).values[..., -1:]    # (B,1) k-th largest
            pm = pbar * (pbar >= kth).to(pbar.dtype)              # keep top-k, zero the rest
            pm = pm / pm.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            B, n_slots, n_ch = logits.shape
            acts = torch.zeros(B, n_slots, dtype=torch.long, device=logits.device)
            acc = torch.zeros(B, n_ch, device=logits.device)
            ar = torch.arange(B, device=logits.device)
            for s in range(n_slots):                               # Webster sequential allocation
                acc = acc + pm
                c = torch.argmax(acc, dim=-1)
                acts[:, s] = c
                acc[ar, c] -= 1.0
            return acts

        if self.stochastic:
            # Sample each per-slot choice from its categorical (Gumbel-max trick keeps
            # this in pure torch without constructing a Distribution object). This spreads
            # routing across DCs the way the trained stochastic policy does, instead of
            # collapsing every slot to the single argmax DC.
            u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(u))
            actions = torch.argmax(logits + gumbel, dim=-1)  # (batch, batch_size)
        else:
            # Greedy: take argmax for each sub-action
            actions = torch.argmax(logits, dim=-1)  # (batch, batch_size)
        return actions

    def _obs_to_batch(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observation dict to batched tensor format for RLModule."""
        import torch

        def to_tensor(v):
            if isinstance(v, dict):
                return {k: to_tensor(vv) for k, vv in v.items()}
            elif isinstance(v, (list, np.ndarray)):
                arr = np.array(v)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                arr = arr[np.newaxis, ...]  # Add batch dim
                # Convert dtype
                if arr.dtype in (np.float64, np.float32):
                    return torch.from_numpy(arr.astype(np.float32))
                elif arr.dtype in (np.int64, np.int32):
                    return torch.from_numpy(arr.astype(np.int64))
                else:
                    return torch.from_numpy(arr)
            elif isinstance(v, (int, float)):
                return torch.tensor([[v]])
            else:
                return v

        return {"obs": to_tensor(obs)}


class DeferringGlobalScheduler(GlobalScheduler):
    """Wraps ANY global scheduler to add the temporal DEFER capability (for a fair
    comparison with the RL's arch-B global defer).

    The inner scheduler picks a target DC for each cloudlet; this wrapper then
    overrides that with DEFER (action index = num_datacenters) when the target DC
    has little green NOW but the forecast says green is coming soon — i.e. hold the
    cloudlet for a greener moment instead of routing it now. This gives the heuristics
    the SAME forecast-driven defer lever as the RL, so the only variable left is the
    routing/defer DECISION (learned vs rule-based). Requires env global_defer_enabled.
    """

    def __init__(self, inner: 'GlobalScheduler', num_datacenters: int, batch_size: int,
                 green_now_thresh: float = 0.05, forecast_thresh: float = 0.3):
        super().__init__(num_datacenters, batch_size)
        self.inner = inner
        # 2026-07-24 scale-fix: the normalized forecast (dc_future_short_mean)
        # for green DCs sits in ~[0, 0.05] on this testbed, so the historical
        # default forecast_thresh=0.3 could NEVER be crossed and the defer rule
        # was inert (every "GQB+defer" run was spatial-only). Thresholds are now
        # env-configurable and the trigger count is tracked for transparency.
        self.green_now_thresh = float(os.environ.get("DEFER_GREEN_NOW_THRESH", green_now_thresh))
        self.forecast_thresh = float(os.environ.get("DEFER_FORECAST_THRESH", forecast_thresh))
        # RELATIVE mode: defer when forecast exceeds current green ratio by margin
        # (scale-free: "greener coming than now"). Off by default.
        self.relative = os.environ.get("DEFER_RELATIVE", "").strip() == "1"
        self.relative_margin = float(os.environ.get("DEFER_RELATIVE_MARGIN", "0.0"))
        self._n_defers = 0
        self._n_calls = 0
        # Track which DCs have EVER shown green this episode. Brown DCs (no turbines)
        # carry a 0.5 PLACEHOLDER forecast that must NOT trigger defer — otherwise
        # cloudlets routed there would be held forever (they never go green).
        self._seen_green = np.zeros(num_datacenters, dtype=bool)

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        actions = list(self.inner.schedule(global_obs))
        nd = self.num_datacenters
        gn = np.asarray(global_obs.get('dc_current_green_power_w', []), dtype=float).ravel()
        gr = np.asarray(global_obs.get('dc_green_ratio', []), dtype=float).ravel()
        fut = np.asarray(global_obs.get('dc_future_short_mean', []), dtype=float).ravel()
        # Update green-capable set from observed green this step.
        for d in range(min(nd, gn.size)):
            if gn[d] > 1.0:
                self._seen_green[d] = True
        for i, dc in enumerate(actions):
            if dc < 0 or dc >= nd:
                continue
            if not self._seen_green[dc]:
                continue  # never-green DC → routing there now is the best we can do
            gr_dc = gr[dc] if dc < gr.size else 0.0
            fut_dc = fut[dc] if dc < fut.size else 0.0
            green_now = ((gn[dc] if dc < gn.size else 0.0) > 1.0
                         or gr_dc > self.green_now_thresh)
            if self.relative:
                # scale-free: forecast promises more green than there is right now
                green_coming = fut_dc > gr_dc + self.relative_margin
            else:
                green_coming = fut_dc > self.forecast_thresh
            if (not green_now) and green_coming:
                actions[i] = nd  # DEFER — hold for the coming green window
                self._n_defers += 1
        self._n_calls += 1
        return actions

    def __str__(self):
        return f"Defer({self.inner})"

    def summary(self) -> str:
        rate = (100.0 * self._n_defers / max(1, self._n_calls))
        return (f"DeferringGlobalScheduler: {self._n_defers} defers over "
                f"{self._n_calls} calls ({rate:.1f}% of slots)")


# === Register all global schedulers ===
GLOBAL_SCHEDULERS = {
    'green_forecast_queue_balanced': GreenForecastQueueBalancedGlobalScheduler,
    'random': RandomGlobalScheduler,
    'round_robin': RoundRobinGlobalScheduler,
    'min_queue': MinQueueGlobalScheduler,
    'green_aware': GreenAwareGlobalScheduler,
    'green_forecast': GreenForecastAwareGlobalScheduler,
    'green_queue_balanced': GreenQueueBalancedGlobalScheduler,
    'min_brown_power': MinBrownPowerGlobalScheduler,
    'weighted_score': WeightedScoreGlobalScheduler,
    'ga': GeneticAlgorithmGlobalScheduler,
    'pso': ParticleSwarmGlobalScheduler,
    'rllib': RLlibGlobalScheduler,  # For Multi-DC (RLlib/Ray) - Old API
    'rllib_new_api': RLlibNewAPIGlobalScheduler,  # For New API Stack (RLModule)
}
