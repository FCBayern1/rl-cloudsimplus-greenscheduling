import logging
import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base import GlobalScheduler

logger = logging.getLogger(__name__)


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


class GreenForecastCapacityGlobalScheduler(GlobalScheduler):
    """Clairvoyant routing with a HARD capacity constraint.

    The existing green_forecast arm sends the batch to whichever site the
    forecast likes best and only nudges for capacity through a soft weight, so
    on the 2026-08-27 headroom probe it piled work onto one site, used 350 Wh of
    green while wasting 7035, and finished 67% of the work. An arm that loses to
    round-robin cannot bound what foresight is worth.

    Here a site drops out of contention once its free PEs are exhausted, and the
    batch falls through to the next-best site that still has room. Desirability
    is future green (dc_future_long_mean, which carries the simulator's own
    future series under green_oracle_mode=godeye) rather than current green.
    """

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        fut = global_obs.get('dc_future_long_mean')
        if fut is None:
            fut = global_obs.get('dc_future_short_mean')
        if fut is None:
            fut = global_obs.get('dc_green_ratio', [0.5] * n)
        des = _as_np_1d(fut, n, fill=0.0, dtype=np.float64)
        free = _as_np_1d(global_obs.get('dc_available_pes'), n, fill=0.0, dtype=np.float64)
        free = np.maximum(free, 0.0).copy()
        order = np.argsort(-des)                      # best forecast first
        actions: List[int] = []
        for _ in range(self.batch_size):
            placed = False
            for dc in order:
                if free[dc] >= 1.0:
                    free[dc] -= 1.0
                    actions.append(int(dc))
                    placed = True
                    break
            if not placed:                             # every site full: least-bad site
                actions.append(int(order[0]))
        return actions


class CurveInformedPlannerGlobalScheduler(GlobalScheduler):
    """Curve-informed feasible planner, spatio-temporal, frozen semantics (Codex 2026-08-30).

    Not an oracle and deliberately not named one. Static and dynamic power are still
    approximations (a fleet draw split by host count, a linear per-PE slope) rather than
    per-host-model curves with idle power-down, so this arm may only be compared against
    blind arms that share the same planner, the same capacity ledger and the same spatial
    carbon model, differing solely in the future information they are given. Calling any
    number from it a headroom measurement requires aligning the power model first and
    checking predicted carbon against simulated carbon digit by digit.

    Two earlier attempts were not oracles and their failures said nothing about
    the testbed. green_forecast ignored capacity and piled the batch onto one
    site. green_forecast_capacity ranked sites by dc_future_long_mean, a level
    statistic, and let the neutral default of the two turbine-free sites carry
    them to the top of the ranking, so 93% of the work went to the dirtiest
    datacentres. Both lost to the blind arm for reasons internal to the arm.

    This one scores a placement by the carbon the job would actually draw:

        J(i,d,s) = sum_tau [ c_g*min(P_i, Gres) + c_b,d*(P_i - Gres)+ ] dt

    over the job's own runtime, against the residual green left after static
    draw and everything already committed. Turbine-free sites enter with G=0 and
    their real brown factor rather than a neutral forecast sentinel. Runtime
    follows CloudSim per-PE semantics, capacity is decremented by the job's PES,
    and the future is wind only: arrivals are not known in advance.

    The green trace is read from the same CSVs the simulator serves, at the
    alignment verified by p0c_step5_alignment.py (offset + 13 + per-DC tz, lag 0
    at r=1.0000 on 18 of 18 cells).

    Three ledgers keep the plan honest. The batch shows only the first 128 queued jobs,
    so a deferred job disappears for long stretches and comes back later. Rebuilding the
    plan every step would lose its reservation and let a later batch book the same future
    capacity twice, which is why reservations persist and are keyed on the stable cloudlet
    id from the evaluator-only planner channel rather than on job shape, which collides.

        scratch          this step's candidate starts, never written to the grid
        reservations     planned but not yet dispatched, keyed by job id, kept across steps
        active           dispatched and executing, released when the run window ends

    occ is the sum of reservations and active and is the only capacity truth. A job that
    already holds a reservation is never replanned, so its plan is stable by construction.
    """

    INFO_SOURCE = "curve"
    ALLOW_DEFER = True
    # Steps of true curve a horizon-limited arm may see. 144 is the TimeCAP pred_len.
    HORIZON_STEPS = 144
    # A planning arm books future capacity. A reactive arm does not: it only ever decides
    # about now, so it holds no reservation and cannot double-book.
    RESERVES = True
    # Measured on this gateway by g1/check_route_visibility_lag.py, three routing events
    # at different steps and different batch sizes, all agreeing exactly. A route issued
    # at step t is executing by the state observed at t+1, and dc_available_pes reports
    # that occupancy 7 observations later still. The ledger is held from t+1 and the
    # sentinel compares against occupancy from t-7, so neither is aligned by taste.
    START_LAG = 1
    AVAIL_REPORT_LAG = 7
    # This family emits DEFER itself and must not be wrapped by the threshold rule in
    # DeferringGlobalScheduler. That wrapper only adds defers and never withdraws one, so
    # it would rewrite a dispatch into a wait while the reservation stayed where it was,
    # leaving a booking nothing will ever collect.
    HANDLES_DEFER = True

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        import csv as _csv, pathlib as _pl, yaml as _yaml
        # The evaluation factory constructs every scheduler with the same two
        # arguments, so the testbed constants are read here rather than passed.
        # ORACLE_OFFSET_ROWS must match the --reset-skip window under test.
        cfg_path = _pl.Path(os.environ.get("EVAL_CONFIG_PATH", "config_C.yml"))
        cfg = _yaml.safe_load(open(cfg_path))
        exp = os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")
        blk = cfg[exp]
        dcs = blk["datacenters"]
        turbines = {d["datacenter_id"]: (d.get("turbine_ids") or []) for d in dcs}
        tz_rows = {d["datacenter_id"]: int(d.get("time_zone_offset_rows", 0)) for d in dcs}
        brown_factors = [d["brown_carbon_factor"] for d in sorted(dcs, key=lambda x: x["datacenter_id"])]
        green_factors = [d.get("green_carbon_factor", 0.01) for d in sorted(dcs, key=lambda x: x["datacenter_id"])]
        divisor = float(blk.get("compressed_power_divisor") or 1500.0)
        offset_rows = int(os.environ.get("ORACLE_OFFSET_ROWS", "0"))
        vm_pe_mips = float(dcs[0].get("vm_pe_mips", 40000))
        # A cloudlet runs at cpu_util of a VM PE, so it occupies the site for
        # length / (mips * cpu_util). Java reads this key with a 0.5 default
        # (SimulationSettings:438) and this experiment never sets it, so every job takes
        # twice its nominal runtime. Measured 2026-08-30 against the simulator's own
        # finish events: 12 of 12 cloudlets gave elapsed/(length/mips) = 2.0166 with a
        # standard deviation of 0.0156 and no dependence on PES, and forcing the key to
        # 0.25 moved the ratio to 4.0201. The same default is used here so the planner
        # prices a job over the window it really occupies.
        cpu_util = float(blk.get("cloudlet_cpu_utilization", 0.5))
        if cpu_util <= 0.0:
            cpu_util = 0.5
        hosts = [sum(v for k, v in d.items() if k.startswith("host_count_"))
                 for d in sorted(dcs, key=lambda x: x["datacenter_id"])]
        tot = float(sum(hosts)) or 1.0
        # Fleet static draw subtracted from every site's green before a job is priced,
        # spread over sites by host count. 332 W is the measured C-regime fleet draw and
        # stays the default so every frozen arm prices exactly as before. It is wrong for
        # any other fleet: on the zero-floor twins (SPEC_ASUS_RS500A_DYN) awake hosts draw
        # 1 W, and with idle power-down the real floor depends on packing. Override with
        # PLANNER_STATIC_TOTAL_W (e.g. "0" for the Level-1 zero-floor spiral).
        static_total = float(os.environ.get("PLANNER_STATIC_TOTAL_W", "332.0"))
        static_w = [static_total * h / tot for h in hosts]
        # Capacity is the real VM PE count, not hosts x 64. The old approximation
        # overstated every site by 4 to 8 percent, which let the planner reserve room
        # that does not exist and pushed the shortfall onto the Java backstop.
        cap_pes = []
        for d in sorted(dcs, key=lambda x: x["datacenter_id"]):
            spe = float(d.get("small_vm_pes", 2))
            cap_pes.append(
                float(d.get("initial_s_vm_count", 0)) * spe
                + float(d.get("initial_m_vm_count", 0)) * spe * float(d.get("medium_vm_multiplier", 2))
                + float(d.get("initial_l_vm_count", 0)) * spe * float(d.get("large_vm_multiplier", 4)))
        horizon = int(os.environ.get("ORACLE_HORIZON", "0"))
        year = int(os.environ.get("ORACLE_YEAR", "2021"))
        # The weather clock, defined once and shared with Java rather than guessed.
        #
        #     weather_row(t) = registered_offset + tz + floor((t - origin) / row_seconds)
        #
        # row_seconds comes from the datacentre's time-scaling mode, exactly as
        # GreenEnergyProvider.getTypicalInterval does: 600 s per row under REAL_TIME,
        # 1 s per row under COMPRESSED. The origin is the simulation clock zero, which is
        # what Java uses, so the two agree by construction.
        #
        # This replaces a hard-coded `warmup = 13`. That constant was the CloudSim start-up
        # cost measured under COMPRESSED, where 13 seconds of VM creation advanced the
        # weather by 13 rows. Under a 600 s row it advances the weather by none, so
        # carrying the 13 forward would have put the planner 13 rows away from the wind
        # the simulator actually serves.
        _mode = str(dcs[0].get("time_scaling_mode", "COMPRESSED")).strip().upper()
        row_seconds = 600.0 if _mode == "REAL_TIME" else 1.0
        weather_origin = float(os.environ.get("PLANNER_WEATHER_ORIGIN_SEC", "0"))
        warmup = int(os.environ.get("PLANNER_WEATHER_WARMUP_ROWS", "0"))
        wind_dir = os.environ.get(
            "ORACLE_WIND_DIR",
            str(cfg_path.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
        wd = _pl.Path(wind_dir)
        self.G = np.zeros((num_datacenters, 20000), dtype=np.float64)
        # Mean of everything strictly before the episode window, per site. A climatology
        # arm calibrates on this and never reads the window under test, which is what
        # keeps it causal.
        self.clim = np.zeros(num_datacenters, dtype=np.float64)
        # Raw per-row series plus each site's first row. The step grid is built at the
        # first decision, when the simulator's own clock is known: the planner counts
        # steps from zero while the clock already stands at the CloudSim start-up cost,
        # and indexing G by the step counter is exactly what the old `warmup = 13`
        # constant was patching over.
        self.G_rows = [np.zeros(1, dtype=np.float64) for _ in range(num_datacenters)]
        self.row_base = [0] * num_datacenters
        for d in range(num_datacenters):
            ts = turbines.get(d) or []
            if not ts:
                continue                      # turbine-free site keeps G = 0
            acc = None
            for t in ts:
                v = np.array([float(x["power_kw"] or 0)
                              for x in _csv.DictReader(open(wd / f"Turbine_{t}_{year}.csv"))])
                acc = v if acc is None else acc + v
            base = offset_rows + warmup + tz_rows[d]
            # Expand rows onto the planning grid on absolute row boundaries: step t reads
            # the row that owns it, so a row that starts mid-decision still ends where the
            # simulator ends it rather than 600 steps after the decision.
            self.G_rows[d] = acc * 1000.0 / divisor
            self.row_base[d] = base
            if base > 0:
                self.clim[d] = float(acc[:base].mean()) * 1000.0 / divisor
        self.cb = np.asarray(brown_factors, dtype=np.float64)
        self.cg = np.asarray(green_factors, dtype=np.float64)
        self.static = np.asarray(static_w, dtype=np.float64)
        # Config arithmetic is only the starting point; the simulator's own count wins
        # at the first step. Kept so the two can be compared and reported.
        self.cap_config = np.asarray(cap_pes, dtype=np.float64)
        self.cap = self.cap_config.copy()
        # Registered effective capacity. Empty string disables the check for a testbed
        # the gate was not preregistered against.
        _exp = os.environ.get("PLANNER_EXPECTED_CAP", "480;384;296;240;144").strip()
        self.expected_cap = (np.array([float(x) for x in _exp.split(";")], dtype=np.float64)
                             if _exp else None)
        if self.expected_cap is not None and self.expected_cap.size != num_datacenters:
            self.expected_cap = None
        self.cpu_util = cpu_util
        # Effective rate at which a cloudlet burns its own length.
        self.mips = float(vm_pe_mips) * cpu_util
        self.row_seconds = row_seconds
        self.weather_origin_sec = weather_origin
        self.weather_warmup_rows = warmup
        # 0 means the only bound on how long a job may wait is its own deadline.
        self.horizon = int(horizon)
        self.T = self.G.shape[1]
        self.dyn_per_pe = (214.0 - 51.4) / 64.0
        # Steps of slack held back from the latest feasible start so the Java deadline
        # backstop never has to fire. deadline_forced_count == 0 is a validity gate,
        # not a metric, so the planner must beat the backstop rather than lean on it.
        # Latest start is derived from the backstop actually in force, not searched.
        # The active rule is the legacy fixed lead: MultiDatacenterSimulationCore fires
        # when now + defer_deadline_slack_sec >= deadline, with no reference to runtime,
        # and Java defaults that slack to 600 s (SimulationSettings:436) which this
        # experiment never overrides. A job must therefore be dispatched before both
        # D - S and D - r. A fixed margin grid of {2..64} steps could never reach 600 and
        # was abandoned rather than widened.
        self.backstop_slack = float(blk.get("defer_deadline_slack_sec", 600.0))
        # The slack is seconds and the planner counts steps. They coincide only while the
        # timestep is one second; TB12 runs at 600 s per step, where treating 600 s as 600
        # steps would overstate the lead by a factor of six hundred.
        self.timestep_sec = float(blk.get("simulation_timestep", 1.0)) or 1.0
        self.backstop_slack_steps = int(np.ceil(self.backstop_slack / self.timestep_sec))
        self.backstop_mode = str(blk.get("defer_deadline_force_mode", "legacy")).strip()
        self.eps = int(os.environ.get("PLANNER_LATEST_START_EPS", "2"))
        # A reservation whose start passed this long ago belongs to a job that left the
        # queue without this planner dispatching it. Bookkeeping hygiene, not inference.
        self.stale_grace = int(os.environ.get("ORACLE_STALE_GRACE", "60"))
        self.max_starts = int(os.environ.get("ORACLE_MAX_STARTS", "256"))
        # Which future this arm may plan against, and whether it may wait at all.
        # Class attributes so a subclass names one arm; the environment overrides only
        # when a sweep needs to vary it without adding a registry entry.
        # Two-timepoint contract (Codex 2026-08-30). Up to the registered decision
        # boundary an arm may wait. After it, waiting is closed for everyone and the
        # remainder drains to whatever each arm already committed to, so a fast arm does
        # not burn idle carbon waiting for a slow one and a slow arm cannot park its tail
        # outside the ledger. 0 disables the boundary.
        self.decision_horizon = int(os.environ.get("PLANNER_DECISION_HORIZON", "0"))
        self.info_source = os.environ.get("PLANNER_INFO_SOURCE", self.INFO_SOURCE)
        self.tail_model = os.environ.get("PLANNER_TAIL_MODEL", "climatology")
        self.horizon_steps = int(os.environ.get(
            "PLANNER_HORIZON_STEPS", str(self.HORIZON_STEPS)))
        self.HORIZON_STEPS = self.horizon_steps
        self.allow_defer = os.environ.get(
            "PLANNER_ALLOW_DEFER", "1" if self.ALLOW_DEFER else "0") == "1"
        self.green_now = np.zeros(num_datacenters, dtype=np.float64)
        self.reset()

    def reset(self):
        """Clear the clock and all three ledgers.

        Without this the base-class no-op left the occupancy grid and the clock from the
        previous episode in place, so every episode after the first planned against a
        stale future.
        """
        # occ is the authoritative PE occupancy grid: active plus reservations, never
        # scratch. Every write goes through _hold/_release so the two dicts and the grid
        # cannot drift apart.
        self.draining = False
        self._cap_calibrated = False
        self._grid_built = False
        self._clock0 = None
        self._rows_touched = {}
        self._row_span = {}
        self._rows_signature = ""
        self._startup_row_shift = None
        self._segments = {}
        self.cap = self.cap_config.copy()
        self.occ = np.zeros((self.num_datacenters, self.T), dtype=np.float64)
        self.active = {}          # job id -> [dc, start, end, pes], dispatched
        self.reservations = {}    # job id -> [dc, start, end, pes], planned, not dispatched
        self.t = 0
        self.n_defer = 0
        self.n_dispatch = 0
        self.n_plan = 0
        self.n_replan_skipped = 0
        self.n_stale_dropped = 0
        self.n_fallback = 0
        self.n_drain_pulled = 0
        self.n_drain_dispatched = 0
        self.n_drain_waited = 0
        # Drift sentinel. occ is what the planner believes is running; the simulator
        # reports what actually is. A gap means jobs did not start when the plan assumed,
        # and no amount of internal consistency makes the ledger true.
        self.drift_abs_max = 0.0
        self.drift_abs_sum = 0.0
        self.drift_n = 0
        self.drift_step_max = -1
        # Per-id closure against the simulator's own execution events. dc_available_pes
        # cannot audit a ledger: it never recovers once a cloudlet finishes. These count
        # disagreements between what the planner committed and what actually ran.
        self.n_unplanned_start = 0      # started without this planner ever dispatching it
        self.unplanned_start_ids = set()
        self.n_wrong_dc = 0             # started somewhere other than the committed site
        self.n_missing_start = 0        # dispatched, never seen to start
        self.n_running_unknown = 0      # executing, absent from the planner's active set
        self.running_pes_over_cap = 0.0
        self.dispatched_at = {}         # id -> (dc, step) for every dispatch
        self.started_ids = set()
        self.unknown_running_ids = set()
        self.running_pes = None
        self._cpu_util_checked = False

    def _hold(self, d, s, e, p):
        self.occ[d, max(0, s):max(0, e)] += p

    def _release(self, d, s, e, p):
        self.occ[d, max(0, s):max(0, e)] -= p

    def _build_step_grid(self, clock_now):
        """Lay the per-row wind series onto the planning grid using the simulator's clock.

        Java resolves the wind row from the absolute simulation clock:
        row = base + floor(clock / row_seconds). The planner counts steps from zero, and
        at the first decision the clock already stands at whatever CloudSim spent creating
        VMs. Under a one-second row that offset is whole rows of weather, which is where
        the hard-coded 13 came from; under a 600 second row it is none. Reading the clock
        makes both cases exact and independent of how long start-up happens to take.
        """
        self._clock0 = clock_now
        # Start-up phase preflight. The registered offset names a first wind row; if
        # CloudSim ever spends a whole row creating VMs, that row silently shifts and the
        # window is no longer the one the artifact registered.
        phase = int(np.floor((clock_now - self.weather_origin_sec) / self.row_seconds))
        self._startup_row_shift = phase
        # Under a one-second row every second of start-up is a whole row, so the shift is
        # unavoidable and is the known defect of that time base rather than a new fault.
        # It is recorded instead of raised, and the physical base is where it must be zero.
        if phase != 0 and self.row_seconds > 1.0:
            raise RuntimeError(
                f"start-up consumed {clock_now - self.weather_origin_sec:.1f}s, which is "
                f"{phase} whole wind row(s) at {self.row_seconds:.0f}s per row. The "
                f"registered offset no longer names the first row this cell will read. "
                f"Refusing to shift the window silently.")
        steps = np.arange(self.T, dtype=np.float64)
        for d in range(self.num_datacenters):
            acc = self.G_rows[d]
            if acc.size <= 1:
                continue
            rows = self.row_base[d] + np.floor(
                (clock_now + steps - self.weather_origin_sec) / self.row_seconds)
            rows = np.clip(rows.astype(np.int64), 0, acc.size - 1)
            self.G[d, :] = acc[rows]
            self._rows_touched[d] = rows
        # Rows actually visited, enumerated from the shared mapping rather than assumed.
        # A 7200 step episode does not touch 7200/600 = 12 rows when the clock starts part
        # way into a row, and the terminal drain does not touch 20; both are computed.
        # The signature covers the whole segment table, (dc, row, seconds spent in it),
        # not just the first and last row. A clock of 13 and a clock of 14 visit the same
        # thirteen rows but weight the first and last differently, and two arms that
        # differ only in that are not running the same window.
        import hashlib as _hl
        parts = []
        for d in range(self.num_datacenters):
            r = self._rows_touched.get(d)
            if r is None or self.G_rows[d].size <= 1:
                continue
            for horizon in (7200, 12000):
                seg = r[:min(horizon, len(r))]
                self._row_span[(d, horizon)] = (int(seg[0]), int(seg[-1]),
                                                int(len(np.unique(seg))))
            seg = r[:min(12000, len(r))]
            rows_u, secs = np.unique(seg, return_counts=True)
            self._segments[d] = list(zip(rows_u.tolist(), secs.tolist()))
            parts.append(f"{d}=" + ",".join(f"{ri}:{sc}" for ri, sc in self._segments[d]))
        self._rows_signature = _hl.sha256("|".join(parts).encode()).hexdigest()[:16]
        self._grid_built = True

    def _tail_level(self, d):
        """Green level assumed beyond the forecast horizon, shared by every arm."""
        if self.tail_model == "climatology":
            return self.clim[d]
        if self.tail_model == "persistence":
            return self.green_now[d]
        if self.tail_model == "zero":
            return 0.0
        raise ValueError(f"unknown tail_model {self.tail_model!r}")

    def _green_view(self, d):
        """The green trace this arm is allowed to plan against at the current step.

        Every arm in the family shares the planner, the capacity ledger and the spatial
        carbon model. This method is the single place they differ, so the gap between
        two of them is the value of the information and nothing else.
        """
        if self.info_source == "curve":
            return self.G[d]
        if self.info_source == "curve_horizon":
            # The true curve for as far as a forecast could ever see, then the frozen
            # causal tail every arm in this family shares. This separates "the predictor
            # is not accurate enough" from "the horizon is too short to matter": if even a
            # perfect forecast truncated to the horizon cannot beat the blind, no
            # predictor can, however good.
            view = np.empty(self.T, dtype=np.float64)
            edge = min(self.T, self.t + self.HORIZON_STEPS)
            view[:edge] = self.G[d, :edge]
            view[edge:] = self._tail_level(d)
            return view
        if self.info_source == "persistence":
            # Causal: the future is assumed to hold whatever green is on the meter now.
            view = np.empty(self.T, dtype=np.float64)
            view[:self.t] = self.G[d, :self.t]
            view[self.t:] = self.green_now[d]
            return view
        if self.info_source == "climatology":
            # Causal: a calibrated level per site, no shape, no future look.
            view = np.empty(self.T, dtype=np.float64)
            view[:self.t] = self.G[d, :self.t]
            view[self.t:] = self.clim[d]
            return view
        raise ValueError(f"unknown info_source {self.info_source!r}")

    def _costs_all(self, d, starts, r, p):
        """Cost of every candidate start at one site, in one pass.

        The scalar form cost one slice per (start, site) pair, which at 128
        slots x ~50 starts x 5 sites x 6105 steps is around two hundred million
        slices and ran four times slower than the blind arm. Cumulative sums
        turn each window into two lookups.
        """
        draw = p * self.dyn_per_pe * self.cpu_util
        gres = np.maximum(0.0, self._green_view(d) - self.static[d]
                          - self.occ[d] * self.dyn_per_pe * self.cpu_util)
        green = np.minimum(draw, gres)
        brown = draw - green
        val = self.cg[d] * green + self.cb[d] * brown
        cs = np.concatenate(([0.0], np.cumsum(val)))
        ends = starts + r
        ok = ends < len(cs)
        out = np.full(len(starts), np.inf)
        out[ok] = cs[ends[ok]] - cs[starts[ok]]
        return out

    def _feasible_all(self, d, starts, r, p):
        """Which candidate starts keep occupied PEs within the real VM capacity."""
        over = (self.occ[d] + p > self.cap[d]).astype(np.int64)
        cs = np.concatenate(([0], np.cumsum(over)))
        ends = starts + r
        ok = ends < len(cs)
        res = np.zeros(len(starts), dtype=bool)
        res[ok] = (cs[ends[ok]] - cs[starts[ok]]) == 0
        return res

    def _cost(self, d, s, r, p):
        return float(self._costs_all(d, np.array([s]), r, p)[0])

    def _plan(self, r, p, latest):
        """Cheapest feasible (site, start) for one job, or None when nothing fits."""
        hi = latest
        if self.horizon > 0:
            hi = min(hi, self.t + self.horizon)
        if hi < self.t + self.START_LAG:
            hi = self.t + self.START_LAG
        lo = self.t + self.START_LAG
        span = max(0, hi - lo)
        stride = max(1, r // 16, (span // self.max_starts) + 1)
        starts = np.arange(lo, hi + 1, stride, dtype=np.int64)
        if len(starts) and starts[-1] != hi:
            starts = np.append(starts, hi)
        starts = starts[starts + r < self.T]
        if not len(starts):
            return None
        best, barg = None, None
        for d in range(self.num_datacenters):
            cost = self._costs_all(d, starts, r, p)
            cost[~self._feasible_all(d, starts, r, p)] = np.inf
            i = int(np.argmin(cost))
            if np.isfinite(cost[i]) and (best is None or cost[i] < best):
                best, barg = float(cost[i]), (d, int(starts[i]))
        return barg

    def _fallback_now(self, r, p):
        """Cheapest site that can take the job immediately, ignoring nothing but choice.

        Reached only when no feasible future start exists inside the deadline. Routing
        now on the cheapest feasible site is strictly better than letting the backstop
        pick, and it is recorded so the run can be judged on how often it happened.
        """
        starts = np.array([self.t + self.START_LAG], dtype=np.int64)
        best, barg = None, None
        for d in range(self.num_datacenters):
            if not bool(self._feasible_all(d, starts, r, p)[0]):
                continue
            c = float(self._costs_all(d, starts, r, p)[0])
            if np.isfinite(c) and (best is None or c < best):
                best, barg = c, d
        if barg is None:
            barg = int(np.argmin(self.cb))
        return int(barg)

    def _runtime_steps(self, length):
        """Steps a cloudlet occupies its site: ceil(length / (mips * u)), PES independent.

        self.mips already carries the utilisation factor. Measured 2026-08-30 against the
        simulator's own finish events over twelve cloudlets: the ratio to length/mips was
        2.0166 with sd 0.0156 at u = 0.5, and 4.0201 at u = 0.25, with no dependence on
        PES in either case.
        """
        return max(1, int(np.ceil(max(0.0, length) / self.mips)))

    @staticmethod
    def _records(csv_text, fields):
        out = []
        if not csv_text:
            return out
        for rec in str(csv_text).split(';'):
            parts = rec.split(':')
            if len(parts) >= fields:
                out.append(parts)
        return out

    def _close_the_books(self, global_obs):
        """Reconcile the planner's commitments with what the simulator actually ran.

        Every start event must belong to a job this planner dispatched, and to the site it
        committed to. Anything else means the backstop or some other path put work on a
        machine the plan never accounted for, and no amount of internal consistency makes
        the resulting carbon attributable.
        """
        # The configured utilisation must be the one the JVM applied. A silent fallback to
        # Java's 0.5 default is exactly how every runtime in this testbed came to be twice
        # its nominal value without anyone noticing for months.
        eff = global_obs.get('cloudlet_cpu_utilization_effective')
        if eff is not None and not self._cpu_util_checked:
            self._cpu_util_checked = True
            if abs(float(eff) - self.cpu_util) > 1e-9:
                raise RuntimeError(
                    f"cloudlet_cpu_utilization mismatch: the planner priced runtimes at "
                    f"{self.cpu_util} but the simulator is applying {float(eff)}. Every "
                    f"runtime, deadline and carbon integral would be wrong by the ratio.")
            eff_slack = global_obs.get('defer_deadline_slack_sec_effective')
            if eff_slack is not None and abs(float(eff_slack) - self.backstop_slack) > 1e-9:
                raise RuntimeError(
                    f"defer_deadline_slack_sec mismatch: the planner aligns latest start "
                    f"to {self.backstop_slack}s but the simulator uses {float(eff_slack)}s. "
                    f"Every deferred job would be force-routed on a lead nobody planned "
                    f"for, which is exactly what the fixed margin grid ran into.")
            eff_mode = global_obs.get('defer_deadline_force_mode_effective')
            if eff_mode is not None and str(eff_mode).strip() != self.backstop_mode:
                raise RuntimeError(
                    f"defer_deadline_force_mode mismatch: the planner assumes "
                    f"{self.backstop_mode!r} but the simulator is running "
                    f"{str(eff_mode).strip()!r}. The two rules bind at different times.")

        started = self._records(global_obs.get('exec_started_csv'), 4)
        running = self._records(global_obs.get('exec_running_csv'), 3)
        run_pes = self._records(global_obs.get('dc_running_pes_csv'), 1)

        for rec in started:
            jid, dc = int(rec[0]), int(rec[1])
            self.started_ids.add(jid)
            committed = self.dispatched_at.get(jid)
            if committed is None:
                self.n_unplanned_start += 1
                self.unplanned_start_ids.add(jid)     # Stage D' P0' per-id closure
            elif committed[0] != dc:
                self.n_wrong_dc += 1

        # Distinct ids, not step occurrences: a single unaccounted job executing for a
        # thousand steps is one problem, not a thousand.
        known = set(self.active) | set(self.dispatched_at)
        for rec in running:
            jid = int(rec[0])
            if jid not in known and jid not in self.unknown_running_ids:
                self.unknown_running_ids.add(jid)
                self.n_running_unknown += 1

        if run_pes:
            vals = np.array([float(x) for x in str(
                global_obs.get('dc_running_pes_csv')).split(',')], dtype=np.float64)
            if vals.size == self.num_datacenters:
                self.running_pes = vals
                over = float(np.max(np.maximum(0.0, vals - self.cap)))
                self.running_pes_over_cap = max(self.running_pes_over_cap, over)

    def _latest_start(self, j, r, present):
        """Last step at which this job can start and still beat the backstop.

            latest = D - max(r + eps, S + eps)

        Short jobs are bound by the fixed 600 s lead, long jobs by their own runtime, and
        no job pays an extra runtime just to dodge the backstop. eps is frozen at 2 steps
        for action quantisation and is not searched.
        """
        if not present[j]:
            return self.T - r - 1
        deadline = self.t + int(np.floor(float(self._ttd[j])))
        if self.backstop_mode == "legacy":
            lead = max(r + self.eps, self.backstop_slack_steps + self.eps)
        else:
            lead = r + self.eps
        return deadline - lead

    def _reactive_choice(self, r, p):
        """Cheapest site whose residual green covers this job right now, else None.

        The stopping rule is causal and books nothing: either there is enough green on
        the meter this instant to run the job without drawing brown, in which case go,
        or there is not, in which case wait and ask again next step.
        """
        draw = p * self.dyn_per_pe * self.cpu_util
        starts = np.array([self.t + self.START_LAG], dtype=np.int64)
        best, barg = None, None
        for d in range(self.num_datacenters):
            if not bool(self._feasible_all(d, starts, r, p)[0]):
                continue
            gres = max(0.0, float(self.green_now[d]) - self.static[d]
                       - float(self.occ[d, int(starts[0])]) * self.dyn_per_pe * self.cpu_util)
            if gres < draw:
                continue
            c = float(self._costs_all(d, starts, r, p)[0])
            if np.isfinite(c) and (best is None or c < best):
                best, barg = c, d
        return barg

    def schedule(self, global_obs):
        n = self.num_datacenters
        planner = global_obs.get('planner')
        if planner is None:
            raise RuntimeError(
                "curve_planner needs the evaluator-only planner channel (stable cloudlet "
                "ids and raw seconds to deadline). It was absent, which means the gateway "
                "predates getBatchCloudletIds or the evaluator did not forward info. "
                "Refusing to plan against filled sentinels: the previous build silently "
                "took time_to_deadline = 1e9 for every slot and never saw a deadline.")
        # The only live signal any arm in this family reads. The curve arm ignores it,
        # the persistence arm plans the whole future on it.
        gn = _as_np_1d(global_obs.get('dc_current_green_power_w'), n, fill=0.0, dtype=np.float64)
        self.green_now = gn
        if not self._grid_built:
            self._build_step_grid(float(planner.get('current_clock', 0.0)))

        self.draining = 0 < self.decision_horizon <= self.t

        # Compare the planner's belief about occupied PEs with the simulator's own count
        # before this step's decisions change either.
        avail = _as_np_1d(global_obs.get('dc_available_pes'), n, fill=np.nan, dtype=np.float64)
        if not self._cap_calibrated and not np.all(np.isnan(avail)):
            # Capacity comes from the simulator, not from arithmetic on the config. A
            # site's usable PEs are min(VM PEs configured, host PEs installed), and on
            # this testbed three of five sites configure more VM PEs than their hosts
            # can carry, so both hosts*64 and the VM total overstate them.
            #
            # Codex 2026-08-30 locked three conditions on this read. It happens once, at
            # an initialisation point with nothing running, it is redone and refrozen
            # after every reset, and the value must match the registered vector or the
            # run stops rather than proceeding on a capacity nobody checked.
            util = _as_np_1d(global_obs.get('dc_utilizations'), n, fill=0.0, dtype=np.float64)
            if float(np.max(util)) > 0.0 or self.t != 0:
                raise RuntimeError(
                    f"planner capacity must be read at an idle initialisation point; "
                    f"found t={self.t} and max utilisation {float(np.max(util)):.4f}. "
                    f"A capacity read against running load silently understates the site.")
            observed_cap = avail + self.occ[:, 0]
            if self.expected_cap is not None and not np.array_equal(
                    observed_cap, self.expected_cap):
                raise RuntimeError(
                    f"effective capacity {observed_cap.tolist()} does not match the "
                    f"registered vector {self.expected_cap.tolist()}. Stopping: the "
                    f"testbed is not the one the gate was preregistered against.")
            if not np.allclose(observed_cap, self.cap_config):
                logger.info(
                    "planner capacity calibrated from the simulator: %s (config arithmetic "
                    "gave %s)", observed_cap.tolist(), self.cap_config.tolist())
            self.cap = observed_cap
            self._cap_calibrated = True
        # Closure runs after the capacity read, so an over-capacity check is made against
        # the simulator's own number rather than the config guess.
        self._close_the_books(global_obs)

        if not np.all(np.isnan(avail)):
            observed = self.cap - avail
            back = self.t - self.AVAIL_REPORT_LAG
            predicted = (self.occ[:, back] if 0 <= back < self.T
                         else np.zeros(self.num_datacenters))
            gap = float(np.max(np.abs(predicted - observed)))
            self.drift_abs_sum += gap
            self.drift_n += 1
            if gap > self.drift_abs_max:
                self.drift_abs_max = gap
                self.drift_step_max = self.t

        ids = np.asarray(planner['batch_cloudlet_ids'], dtype=np.int64)
        mi = np.asarray(planner['batch_cloudlet_mi'], dtype=np.float64)
        pes = np.asarray(planner['batch_cloudlet_pes'], dtype=np.float64)
        ttd = np.asarray(planner['batch_cloudlet_time_to_deadline'], dtype=np.float64)
        present = np.asarray(planner['batch_cloudlet_deadline_present'], dtype=np.int64)
        self._ttd = ttd

        # Shadow budget for this step: what the simulator reports free, less whatever
        # this step has already committed. Every dispatch draws it down, so a reservation
        # coming due and a drained backlog job compete for the same real PEs.
        # Headroom is capacity less what is genuinely executing, from the simulator's own
        # running-PE count. The earlier version used dc_available_pes, which is a VM
        # allocation counter that never recovers, so it shrank monotonically and would
        # have throttled the drain to nothing.
        if self.running_pes is not None:
            budget = (self.cap - self.running_pes).astype(np.float64)
        else:
            budget = (self.cap - self.occ[:, min(self.t, self.T - 1)]).astype(np.float64)
        drain_queue = []

        # Retire executions that have finished and reservations whose start went by
        # without this planner dispatching them, which means the job left the queue
        # some other way.
        for jid, (d, s, e, pp) in list(self.active.items()):
            if e <= self.t:
                self._release(d, s, e, pp)
                del self.active[jid]
        for jid, (d, s, e, pp) in list(self.reservations.items()):
            if self.t > s + self.stale_grace:
                self._release(d, s, e, pp)
                del self.reservations[jid]
                self.n_stale_dropped += 1

        actions = []
        for j in range(self.batch_size):
            jid = int(ids[j])
            if jid < 0:
                actions.append(n)          # padding slot, never planned, never routed
                continue
            if jid in self.active:
                actions.append(n)          # already executing, must not be routed twice
                continue

            p = max(1.0, float(pes[j]))
            r = self._runtime_steps(float(mi[j]))

            if not self.RESERVES:
                # Reactive stopping rule: decide about now, never about later.
                if self.allow_defer and not self.draining:
                    latest = self._latest_start(j, r, present)
                else:
                    latest = self.t
                d = self._reactive_choice(r, p)
                if d is None and self.t < latest:
                    actions.append(n)                     # not green enough yet, wait
                    self.n_defer += 1
                    continue
                if d is None:
                    d = self._fallback_now(r, p)          # margin reached, go regardless
                    self.n_fallback += 1
                start = self.t + self.START_LAG
                e = start + r
                self._hold(d, start, e, p)
                self.active[jid] = (d, start, e, p)
                budget[d] -= p
                self.dispatched_at[jid] = (d, self.t)
                actions.append(int(d))
                self.n_dispatch += 1
                self.n_plan += 1
                continue

            held = self.reservations.get(jid)
            if held is not None:
                self.n_replan_skipped += 1
                d, s, e, pp = held
                # A reservation frozen before the boundary is a commitment, not a
                # decision still open. It keeps its site and its start. Pulling four
                # thousand of them forward at the boundary destroyed a feasible plan and
                # oversubscribed every site at once, which is why it is gone.
                if s <= self.t + self.START_LAG:
                    del self.reservations[jid]
                    self.active[jid] = (d, s, e, pp)
                    budget[d] -= pp
                    self.dispatched_at[jid] = (d, self.t)
                    actions.append(int(d))
                    self.n_dispatch += 1
                else:
                    actions.append(n)
                    self.n_defer += 1
                continue

            if self.draining:
                # Past the boundary nothing new is planned and nothing is re-optimised.
                # Unreserved backlog leaves through a drain shared by every arm, rate
                # limited by the capacity the simulator actually reports free this step
                # minus what this step has already committed, and ordered by latest
                # start so the tightest deadline leaves first. A job that does not fit
                # waits for the next step rather than being forced out.
                drain_queue.append((self._latest_start(j, r, present), j, jid, r, p))
                actions.append(n)
                continue

            # A job with no deadline may wait as long as the horizon allows; one with a
            # deadline must start early enough to finish before it, minus the margin
            # that keeps the Java backstop out of the loop.
            if not self.allow_defer:
                latest = self.t                        # no-wait arm: spatial choice only
            else:
                latest = self._latest_start(j, r, present)
            barg = self._plan(r, p, latest)
            if barg is None:
                d = self._fallback_now(r, p)
                s = self.t
                self.n_fallback += 1
            else:
                d, s = barg
            e = s + r
            self.n_plan += 1
            self._hold(d, s, e, p)
            if s <= self.t + self.START_LAG:
                self.active[jid] = (d, s, e, p)
                budget[d] -= p
                self.dispatched_at[jid] = (d, self.t)
                actions.append(int(d))
                self.n_dispatch += 1
            else:
                self.reservations[jid] = (d, s, e, p)
                actions.append(n)                                  # DEFER
                self.n_defer += 1

        # Shared feasibility drain. Tightest latest start goes first; a job leaves only
        # if the site the shared cost model prefers still has real headroom this step,
        # otherwise it waits. No fixed job count, no aggregate back-inference, and the
        # site choice is the same current-cost model every arm uses.
        for _latest, j, jid, r, p in sorted(drain_queue):
            start = self.t + self.START_LAG
            starts = np.array([start], dtype=np.int64)
            best, barg = None, None
            for d in range(n):
                if budget[d] < p:
                    continue
                if not bool(self._feasible_all(d, starts, r, p)[0]):
                    continue
                c = float(self._costs_all(d, starts, r, p)[0])
                if np.isfinite(c) and (best is None or c < best):
                    best, barg = c, d
            if barg is None:
                self.n_drain_waited += 1
                continue                       # no headroom this step, try again next
            e = start + r
            self._hold(barg, start, e, p)
            self.active[jid] = (barg, start, e, p)
            budget[barg] -= p
            self.dispatched_at[jid] = (barg, self.t)
            actions[j] = int(barg)
            self.n_dispatch += 1
            self.n_drain_dispatched += 1

        self.t += 1
        return actions

    def metrics(self) -> Dict[str, float]:
        """Counters the validity contract is checked against, for the results row."""
        return {
            "planner_info_source": self.info_source,
            "planner_tail_model": self.tail_model,
            "planner_horizon_steps": self.horizon_steps,
            "planner_allow_defer": int(self.allow_defer),
            "planner_reserves": int(self.RESERVES),
            "planner_backstop_slack": self.backstop_slack,
            "planner_backstop_slack_steps": self.backstop_slack_steps,
            "planner_timestep_sec": self.timestep_sec,
            "planner_row_seconds": self.row_seconds,
            "planner_weather_origin_sec": self.weather_origin_sec,
            "planner_weather_warmup_rows": self.weather_warmup_rows,
            # Hidden pricing quantities, reported so a verdict can fail-fast on a run
            # whose environment drifted from the registered one (HZ prereg, G0).
            "planner_static_total_w": float(self.static.sum()),
            "planner_expected_cap": (";".join(f"{int(x)}" for x in self.expected_cap)
                                     if self.expected_cap is not None else ""),
            "planner_clock0": self._clock0,
            "planner_startup_row_shift": self._startup_row_shift,
            "planner_rows_signature": self._rows_signature,
            "planner_rows_7200": ";".join(
                f"{d}:{v[0]}-{v[1]}({v[2]})" for (d, h), v in sorted(self._row_span.items())
                if h == 7200),
            "planner_rows_12000": ";".join(
                f"{d}:{v[0]}-{v[1]}({v[2]})" for (d, h), v in sorted(self._row_span.items())
                if h == 12000),
            "planner_backstop_mode": self.backstop_mode,
            "planner_latest_start_eps": self.eps,
            "planner_cpu_util": self.cpu_util,
            "planner_effective_mips": self.mips,
            "planner_decision_horizon": self.decision_horizon,
            "planner_n_plan": self.n_plan,
            "planner_n_dispatch": self.n_dispatch,
            "planner_n_defer": self.n_defer,
            "planner_n_fallback": self.n_fallback,
            "planner_n_drain_pulled": self.n_drain_pulled,
            "planner_n_drain_dispatched": self.n_drain_dispatched,
            "planner_n_drain_waited": self.n_drain_waited,
            "planner_n_stale_dropped": self.n_stale_dropped,
            "planner_open_reservations": len(self.reservations),
            "planner_active": len(self.active),
            "planner_occ_max_over_cap": float(np.max(
                np.maximum(0.0, self.occ.max(axis=1) - self.cap))),
            "planner_drift_abs_max": self.drift_abs_max,
            "planner_drift_abs_mean": (self.drift_abs_sum / self.drift_n) if self.drift_n else 0.0,
            "planner_drift_step_max": self.drift_step_max,
            "planner_n_unplanned_start": self.n_unplanned_start,
            # sorted ids of the unplanned starts, and their sha256, so a P0' contract can
            # check they are exactly the jobs the deadline mask re-routed (design §16)
            "planner_unplanned_start_ids": ";".join(str(i) for i in sorted(self.unplanned_start_ids)),
            "planner_unplanned_start_ids_sha": __import__("hashlib").sha256(
                ";".join(str(i) for i in sorted(self.unplanned_start_ids)).encode()).hexdigest()[:16],
            "planner_n_wrong_dc": self.n_wrong_dc,
            "planner_n_running_unknown": self.n_running_unknown,
            "planner_n_dispatched_never_started": len(
                set(self.dispatched_at) - self.started_ids),
            "planner_running_pes_over_cap": self.running_pes_over_cap,
            "planner_cap_observed": ";".join(f"{c:.0f}" for c in self.cap),
            "planner_cap_config": ";".join(f"{c:.0f}" for c in self.cap_config),
        }

    def summary(self) -> str:
        return (f"CurveInformedPlannerGlobalScheduler: planned {self.n_plan}, dispatched "
                f"{self.n_dispatch}, deferred {self.n_defer}, honoured existing "
                f"reservations {self.n_replan_skipped}, fell back to route-now "
                f"{self.n_fallback}, pulled forward at the boundary {self.n_drain_pulled}, "
                f"drained {self.n_drain_dispatched} (held back {self.n_drain_waited}), "
                f"dropped stale reservations {self.n_stale_dropped}; "
                f"open reservations {len(self.reservations)}, active {len(self.active)}; "
                f"occupancy drift max {self.drift_abs_max:.1f} PEs at step "
                f"{self.drift_step_max}, mean "
                f"{(self.drift_abs_sum / self.drift_n) if self.drift_n else 0.0:.2f}")


class HorizonLimitedOraclePlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """A perfect forecast truncated to the horizon a real predictor has, then a blind tail.

    Codex 2026-08-31: this is the gate that separates the two ways the TimeCAP arm could
    fail. If a perfect 144 step view plus the shared causal tail already cannot beat the
    blind by five percent, the horizon itself is too short and no predictor can pass,
    however accurate. Only if this clears is it worth wiring the real forecast in.
    """

    INFO_SOURCE = "curve_horizon"


class PersistencePlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Same planner, same ledgers, same carbon model. The future is assumed to look like
    the meter reading right now, which is the standard causal no-forecast reference."""

    INFO_SOURCE = "persistence"


class ClimatologyPlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Same planner. The future is a per-site level calibrated on the history strictly
    before the episode window, so it knows the climate but not the weather."""

    INFO_SOURCE = "climatology"


class ReactiveWaitPlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Wait for green rather than plan for it, on the shared spatial cost model.

    Codex 2026-08-30: persistence is not this arm. Flattening the future to the current
    level leaves every candidate start equally priced, so persistence takes the earliest
    one and behaves like immediate planning. This arm is a different causal rule. It
    books no future capacity, it asks each step whether the meter already carries the
    job, and it routes unconditionally once the frozen latest-start margin arrives.
    Naive wait-for-green has been the strongest blind before, so leaving it out would
    make the strongest causal blind incomplete.
    """

    INFO_SOURCE = "persistence"
    RESERVES = False


class NoWaitPlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Same planner and the same spatial carbon model, with the temporal lever removed.

    The gap to a waiting arm is the value of waiting; the gap between two waiting arms
    with different green views is the value of the forecast. Keeping them separate is
    what the first smoke failed to do, since its blind arm could not wait at all.
    """

    INFO_SOURCE = "persistence"
    ALLOW_DEFER = False


class PerturbedOraclePlannerGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """The oracle144 arm with its eyes degraded by a registered corruption tier.

    Same planner, same ledgers, same tail, same carbon model as the whole family; the
    only change is that the next HORIZON_STEPS rows of the green view pass through the
    frozen perturbation ladder in `forecast_perturb`. The gap to `oracle144_planner` is
    therefore the cost of that quality loss and nothing else, and the curve over tiers is
    the dose-response of forecast value in this scenario. Settlement always uses the true
    trace; only the planning view is corrupted.

    Tier comes from PLANNER_PERTURB_TIER (default godeye, which must equal oracle144
    bit for bit). Tier timecap_cal additionally needs PLANNER_PERTURB_CAL pointing at the
    calibration artifact measured on the existing TimeCAP checkpoint's residuals.
    """

    INFO_SOURCE = "curve_horizon_perturbed"

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        from . import forecast_perturb as _fp
        self._fp = _fp
        import json as _json
        self.perturb_tier = os.environ.get("PLANNER_PERTURB_TIER", "godeye")
        # Ladder-v2 (Codex 2026-09-02): lead 0 is an observation in every tier, and the
        # calibrated tier is the DC-level one-factor surrogate. v1 stays available so the
        # recorded A-prime run remains reproducible bit for bit.
        self.perturb_v2 = os.environ.get("PLANNER_PERTURB_V2", "0") == "1"
        # Scheme 2-E mode: the E tier set, with the audit-calibrated primary error.
        self.perturb_e = os.environ.get("PLANNER_PERTURB_E", "0") == "1"
        if self.perturb_e:
            tiers = _fp.TIERS_E
        else:
            tiers = _fp.TIERS_V2 if self.perturb_v2 else _fp.TIERS
        if os.environ.get("PLANNER_PERTURB_PILOT", "0") == "1" and self.perturb_v2:
            tiers = {**tiers, **_fp.TIERS_PILOT}
        if self.perturb_tier not in tiers:
            raise ValueError(f"unknown perturb tier {self.perturb_tier!r}; "
                             f"registered: {sorted(tiers)}")
        cal_path = os.environ.get("PLANNER_PERTURB_CAL", "")
        if cal_path and self.perturb_e:
            audit = _json.load(open(cal_path))
            self.perturb_calibration = audit["primary_error_params"]
        elif cal_path and self.perturb_v2:
            self.perturb_calibration = _json.load(open(cal_path))
        elif cal_path:
            self.perturb_calibration = _fp.load_calibration(cal_path)
        else:
            self.perturb_calibration = None
        if self.perturb_tier == "timecap_cal" and self.perturb_calibration is None:
            raise ValueError("tier timecap_cal needs PLANNER_PERTURB_CAL")
        if self.perturb_tier == "checkpoint_residual_surrogate_v2" \
                and self.perturb_calibration is None:
            raise ValueError("the surrogate tier needs PLANNER_PERTURB_CAL")
        if self.perturb_tier == "calibrated_shrink_v1" \
                and self.perturb_calibration is None:
            raise ValueError("calibrated_shrink_v1 needs PLANNER_PERTURB_CAL "
                             "(the error-audit json)")

    def _green_view(self, d):
        if self.info_source != "curve_horizon_perturbed":
            return super()._green_view(d)
        # Measured past, corrupted forecast window, then the frozen causal tail every arm
        # in the family shares, exactly as curve_horizon splices its perfect window.
        view = np.empty(self.T, dtype=np.float64)
        view[:self.t] = self.G[d, :self.t]
        edge = min(self.T, self.t + self.HORIZON_STEPS)
        if self.perturb_v2 or self.perturb_e:
            if getattr(self, "_episode_key_T", None) != self.T:
                import hashlib as _hl
                self._episode_key = _hl.sha256(
                    np.ascontiguousarray(self.G, dtype=np.float64).tobytes()).hexdigest()
                self._episode_key_T = self.T
            if self.perturb_e:
                fut = self._fp.perturbed_future_e(
                    self.G[d], self.t, self.HORIZON_STEPS, d, self.perturb_tier,
                    eparams=(self.perturb_calibration
                             if self.perturb_tier == "calibrated_shrink_v1" else None),
                    common_key=self._episode_key)
            else:
                fut = self._fp.perturbed_future_v2(
                    self.G[d], self.t, self.HORIZON_STEPS, d, self.perturb_tier,
                    self.perturb_calibration, common_key=self._episode_key)
        else:
            fut = self._fp.perturbed_future(
                self.G[d], self.t, self.HORIZON_STEPS, d, self.perturb_tier,
                self.perturb_calibration)
        view[self.t:edge] = fut[:edge - self.t]
        view[edge:] = self._tail_level(d)
        return view


class LoadSmoothingGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Forecast-free load smoothing (Scheme 2-E blind family, Codex 2026-09-02).

    Same chassis, ledgers and commitment semantics as the whole family, but a slot is
    priced ONLY by the reservation-ledger occupancy it would overlap, ties to the
    earliest start; queue, capacity and deadlines are the only inputs. The arm never
    reads a green value, present or future, and a test drives a decision with the green
    view rigged to explode to prove it. Its role is adversarial: if spreading alone
    captures the forecast arms' benefit, the clean-vs-blind gate must fail honestly.
    """

    INFO_SOURCE = "occupancy_only"

    def _green_view(self, d):
        raise RuntimeError("load_smoothing must never price a slot from green")

    def _tail_level(self, d):
        raise RuntimeError("load_smoothing must never price a slot from green")

    def _costs_all(self, d, starts, r, p):
        occ = self.occ[d]
        cs = np.concatenate(([0.0], np.cumsum(occ)))
        ends = starts + r
        ok = ends < len(cs)
        out = np.full(len(starts), np.inf)
        overlap = cs[np.minimum(ends, len(cs) - 1)] - cs[np.minimum(starts, len(cs) - 1)]
        # Tie-break to the earliest start with a strictly monotone epsilon so equal
        # overlap never leaves the choice to argmin ordering accidents.
        out[ok] = overlap[ok] + 1e-9 * starts[ok]
        return out

    def _reactive_choice(self, r, p):
        raise RuntimeError("load_smoothing has no reactive green rule")


class ReservationEDFGlobalScheduler(CurveInformedPlannerGlobalScheduler):
    """Online reservation-EDF (Scheme 2-E blind family, Codex 2026-09-02).

    The TB13 A.4 contract policy transplanted onto the online chassis: on arrival a job
    takes the EARLIEST start any site can hold on the reservation ledger, ties to the
    lower site index, and the reservation is irrevocable like every arm's. No green value
    is ever read.
    """

    INFO_SOURCE = "earliest_feasible"

    def _green_view(self, d):
        raise RuntimeError("reservation_edf must never price a slot from green")

    def _tail_level(self, d):
        raise RuntimeError("reservation_edf must never price a slot from green")

    def _costs_all(self, d, starts, r, p):
        occ = self.occ[d]
        cs = np.concatenate(([0.0], np.cumsum(occ)))
        ends = starts + r
        ok = ends < len(cs)
        out = np.full(len(starts), np.inf)
        # Earliest start dominates; the site index separates exact ties so the lower
        # site wins, matching the registered EDF tie order.
        out[ok] = starts[ok].astype(np.float64) + 1e-6 * d
        return out

    def _reactive_choice(self, r, p):
        raise RuntimeError("reservation_edf has no reactive green rule")


class AlwaysDeferGlobalScheduler(GlobalScheduler):
    """Defers every slot, so the deadline backstop becomes the only thing that routes.

    Diagnostic only. It exists to exercise the legacy fixed-lead backstop, which an arm
    that routes on arrival never touches: an A/B run with green_queue_balanced showed
    deadline_forced_count = 0 on both sides and therefore proved nothing about the slack
    or the mode. Under this arm every job is force-routed at D - S, so the two can be
    compared where it matters.
    """

    HANDLES_DEFER = True

    def schedule(self, global_obs):
        return [self.num_datacenters] * self.batch_size


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
class OptionPlannerMixin:
    """Option-mode front end for a CurveInformedPlanner arm (reports/OPTION_ACTION_DESIGN.md §5).

    The base planner decides per slot with its own green view exactly as before: dispatch
    now to d, or wait. In option mode a wait becomes HOLD_FOR_GREEN(site): the reserved
    site for a reserving arm, the cheapest feasible site now for the reactive arm (and for
    a drain-phase wait). The held job never comes back, so the base's reservation is
    turned into a permanent occupancy at its planned start, the best model the arm has of
    when the executor will release it; the executor, not the planner, decides the real
    start (T1 green / T2 fallback). A HOLD the env would mask (deadline or capacity) is
    moved to the cheapest allowed site, and to ROUTE_NOW(site) when none is allowed, so an
    analytic arm never emits an illegal hold.
    """

    OPTION = True

    def schedule(self, global_obs):
        n = self.num_datacenters
        planner = global_obs.get('planner') or {}
        ids = np.asarray(planner.get('batch_cloudlet_ids', []), dtype=np.int64)
        pes = np.asarray(planner.get('batch_cloudlet_pes', []), dtype=np.float64)
        mi = np.asarray(planner.get('batch_cloudlet_mi', []), dtype=np.float64)
        mask = global_obs.get('batch_cloudlet_hold_allowed')
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        t_decide = self.t
        base = super().schedule(global_obs)          # advances self.t
        out = []
        for j, a in enumerate(base):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0 or a < n:
                out.append(int(a) if a < n else 0)   # dispatch now, or padding
                continue
            p = max(1.0, float(pes[j]))
            r = self._runtime_steps(float(mi[j]))
            held = self.reservations.pop(jid, None)
            if held is not None:
                d, s, e, pp = held
                site = int(d)
                self.active[jid] = (d, s, e, pp)      # occupancy already on the grid
            else:
                # cheapest feasible site at the decision step (self.t already advanced)
                start = t_decide + self.START_LAG
                starts = np.array([start], dtype=np.int64)
                best, site = None, None
                for d in range(n):
                    if not bool(self._feasible_all(d, starts, r, p)[0]):
                        continue
                    c = float(self._costs_all(d, starts, r, p)[0])
                    if np.isfinite(c) and (best is None or c < best):
                        best, site = c, d
                if site is None:
                    site = int(np.argmin(self.cb))
                self._hold(site, start, start + r, p)
                self.active[jid] = (site, start, start + r, p)
            self.dispatched_at[jid] = (site, t_decide)
            allowed = None if mask is None or j >= mask.shape[0] else mask[j]
            if allowed is not None and allowed[site] < 0.5:
                cands = [d for d in range(n) if allowed[d] >= 0.5]
                if cands:
                    starts = np.array([t_decide + self.START_LAG], dtype=np.int64)
                    costs = [float(self._costs_all(d, starts, r, p)[0]) for d in cands]
                    site = int(cands[int(np.argmin(costs))])
                else:
                    out.append(site)                  # nothing allowed: ROUTE_NOW(site)
                    self.n_fallback += 1
                    continue
            out.append(n + site)                      # HOLD_FOR_GREEN(site)
        return out


class OraclePlannerOptionGlobalScheduler(OptionPlannerMixin, CurveInformedPlannerGlobalScheduler):
    """oracle_opt: the truth-curve reserving planner deciding ROUTE_NOW / HOLD once per job."""


class PerturbedOraclePlannerOptionGlobalScheduler(OptionPlannerMixin, PerturbedOraclePlannerGlobalScheduler):
    """shuffle_opt / anti_opt / calibrated arms: the perturbed curve through the same rule."""


class PersistencePlannerOptionGlobalScheduler(OptionPlannerMixin, PersistencePlannerGlobalScheduler):
    """persistence_opt (blind): flat future at the current level."""


class ClimatologyPlannerOptionGlobalScheduler(OptionPlannerMixin, ClimatologyPlannerGlobalScheduler):
    """climatology_opt (blind): the mean curve."""


class ReactiveWaitPlannerOptionGlobalScheduler(OptionPlannerMixin, ReactiveWaitPlannerGlobalScheduler):
    """reactive_opt (blind): ROUTE_NOW when the meter covers the job, else HOLD at the
    cheapest feasible site now."""


def offset_action(d, kappa, grid, num_dcs):
    return int(d) * len(grid) + list(grid).index(int(kappa))


def largest_legal_offset(mask_row, d, kappa_target, grid):
    """Largest κ in the grid, not above kappa_target, that the (slot, site, κ) legality row
    allows at site d; None when none is legal (Addendum C3: the analytic arm chooses, the
    executor never clips). Pure."""
    K = len(grid)
    best = None
    for i, k in enumerate(grid):
        if k > kappa_target:
            break
        if mask_row is None or float(mask_row[d * K + i]) >= 0.5:
            best = k
    return best


class OffsetPlannerMixin:
    """(DC, dispatch-offset) front end for a CurveInformedPlanner arm (OPTION_ACTION_DESIGN
    §8, Addenda A5, C2, C3).

    The base planner decides with its own green view: a dispatch now to d, or a reservation
    (d, s). A reservation becomes (d, κ) with κ the largest legal grid offset not above
    s - t - lag (quantised down); a reactive-style wait (no reservation) becomes the largest
    legal offset at the cheapest feasible site by current visible cost. If no positive offset
    is legal at that site the job is dispatched now there. The base's reservation is kept on
    its grid as the arm's model of the executor's start.
    """

    OPTION = True

    def _grid(self, global_obs):
        g = global_obs.get("offset_grid")
        if g is None:
            from gym_cloudsimplus.envs.option_executor import offset_grid
            g = offset_grid(int(os.environ.get("OFFSET_WAIT_CAP_STEPS", "72")))
        return list(g)

    def _cheapest_site(self, start, r, p):
        starts = np.array([start], dtype=np.int64)
        best, site = None, None
        for d in range(self.num_datacenters):
            if not bool(self._feasible_all(d, starts, r, p)[0]):
                continue
            c = float(self._costs_all(d, starts, r, p)[0])
            if np.isfinite(c) and (best is None or c < best):
                best, site = c, d
        return int(np.argmin(self.cb)) if site is None else site

    def schedule(self, global_obs):
        n = self.num_datacenters
        grid = self._grid(global_obs)
        planner = global_obs.get('planner') or {}
        ids = np.asarray(planner.get('batch_cloudlet_ids', []), dtype=np.int64)
        pes = np.asarray(planner.get('batch_cloudlet_pes', []), dtype=np.float64)
        mi = np.asarray(planner.get('batch_cloudlet_mi', []), dtype=np.float64)
        mask = global_obs.get('batch_cloudlet_offset_allowed')
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        t_decide = self.t
        base = super().schedule(global_obs)          # advances self.t
        out = []
        for j, a in enumerate(base):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0:
                out.append(0)
                continue
            row = None if mask is None or j >= mask.shape[0] else mask[j]
            p = max(1.0, float(pes[j]))
            r = self._runtime_steps(float(mi[j]))
            if a < n:
                out.append(offset_action(a, 0, grid, n))          # dispatch now
                continue
            held = self.reservations.pop(jid, None)
            if held is not None:
                d, s, e, pp = held
                site, target = int(d), max(0, int(s) - t_decide - self.START_LAG)
                self.active[jid] = (d, s, e, pp)              # the base's own model of the start
                kappa = largest_legal_offset(row, site, target, grid)
            else:
                site = self._cheapest_site(t_decide + self.START_LAG, r, p)
                kappa = largest_legal_offset(row, site, grid[-1], grid)
                start = t_decide + (kappa or 0) + self.START_LAG
                self._hold(site, start, start + r, p)
                self.active[jid] = (site, start, start + r, p)
            self.dispatched_at[jid] = (site, t_decide)
            if kappa is None:
                kappa = 0
                self.n_fallback += 1
            out.append(offset_action(site, kappa, grid, n))
        return out


class OraclePlannerOffsetGlobalScheduler(OffsetPlannerMixin, CurveInformedPlannerGlobalScheduler):
    """oracle_off: truth-curve reserving planner, planned start quantised down to the grid."""


class PerturbedOraclePlannerOffsetGlobalScheduler(OffsetPlannerMixin, PerturbedOraclePlannerGlobalScheduler):
    """shuffle_off / anti_off (and godeye through the tier): the perturbed curve, same rule."""


class PersistencePlannerOffsetGlobalScheduler(OffsetPlannerMixin, PersistencePlannerGlobalScheduler):
    """persistence_off (blind): flat future at the current level."""


class ClimatologyPlannerOffsetGlobalScheduler(OffsetPlannerMixin, ClimatologyPlannerGlobalScheduler):
    """climatology_off (blind): the mean curve."""


class ReactiveWaitPlannerOffsetGlobalScheduler(OffsetPlannerMixin, ReactiveWaitPlannerGlobalScheduler):
    """reactive_off (blind, C3): dispatch now where the meter covers the job, else the
    largest legal offset at the cheapest feasible site by current visible cost."""


class FixedOffsetGlobalScheduler(OffsetPlannerMixin, PersistencePlannerGlobalScheduler):
    """fixed_off(κ) (blind, C3): every job takes the largest legal offset not above the
    configured κ (FIXED_OFF_KAPPA), at the site of lowest current visible cost for that
    start under the persistence view; κ = 0 is the no-wait arm, κ = W the latest-legal one.
    Reads no curve beyond the current meter."""

    def schedule(self, global_obs):
        n = self.num_datacenters
        grid = self._grid(global_obs)
        kappa_cfg = int(os.environ.get("FIXED_OFF_KAPPA", "0"))
        if kappa_cfg not in grid:
            raise RuntimeError(f"FIXED_OFF_KAPPA={kappa_cfg} is not on the grid {grid}")
        planner = global_obs.get('planner') or {}
        ids = np.asarray(planner.get('batch_cloudlet_ids', []), dtype=np.int64)
        pes = np.asarray(planner.get('batch_cloudlet_pes', []), dtype=np.float64)
        mi = np.asarray(planner.get('batch_cloudlet_mi', []), dtype=np.float64)
        mask = global_obs.get('batch_cloudlet_offset_allowed')
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        # The persistence base keeps the grid / clock / closure bookkeeping and makes its own
        # commitment per slot (a dispatch held from t + lag, or a reservation). That
        # commitment is undone below before this arm books its own, so the grid never
        # carries a job twice.
        t_decide = self.t
        base = PersistencePlannerGlobalScheduler.schedule(self, dict(global_obs))
        out = []
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0:
                out.append(0)
                continue
            row = None if mask is None or j >= mask.shape[0] else mask[j]
            p = max(1.0, float(pes[j]))
            r = self._runtime_steps(float(mi[j]))
            self._undo_base_commitment(jid, t_decide)
            best = None
            for d in range(n):
                k = largest_legal_offset(row, d, kappa_cfg, grid)
                if k is None:
                    continue
                start = t_decide + k + self.START_LAG
                c = float(self._costs_all(d, np.array([start], dtype=np.int64), r, p)[0])
                if np.isfinite(c) and (best is None or (k, -c) > (best[0], -best[1])):
                    best = (k, c, d)
            if best is None:
                out.append(offset_action(int(np.argmin(self.cb)), 0, grid, n))
                self.n_fallback += 1
                continue
            k, _c, d = best
            start = t_decide + k + self.START_LAG
            self._hold(d, start, start + r, p)
            self.active[jid] = (d, start, start + r, p)
            self.dispatched_at[jid] = (d, t_decide)
            out.append(offset_action(d, k, grid, n))
        return out

    def _undo_base_commitment(self, jid, t_decide):
        """Release whatever the base planner booked for this job on this decision step."""
        held = self.reservations.pop(jid, None)
        if held is not None:
            d, s, e, pp = held
            self._release(d, s, e, pp)
            return
        act = self.active.get(jid)
        if act is not None and self.dispatched_at.get(jid, (None, None))[1] == t_decide:
            d, s, e, pp = act
            self._release(d, s, e, pp)
            del self.active[jid]


class ScheduleReplayGlobalScheduler(GlobalScheduler):
    """Replays a fixed schedule {cloudlet id: [site, start step]} (SCHEDULE_JSON) through the
    (DC, dispatch-offset) executor (ERROR_LADDER_PLANNER_PREREG §2.2): at a job's first
    sighting at step t it emits (site, κ) with κ = start − t − lag so the job is routed at
    start − lag and begins at `start`; a start already in the past or a (site, κ) the
    legality mask refuses is emitted as κ = 0 and counted (`n_late`, `n_masked`), never
    clipped silently. Reads no green, no forecast.
    """

    HANDLES_DEFER = True
    OPTION = True
    START_LAG = 1

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        import json as _json
        path = os.environ.get("SCHEDULE_JSON", "").strip()
        if not path:
            raise RuntimeError("schedule_replay needs SCHEDULE_JSON")
        raw = _json.load(open(path))
        self.plan = {int(k): (int(v[0]), int(v[1])) for k, v in raw["schedule"].items()}
        self.grid = list(raw.get("grid") or [])
        self.t = 0
        self.seen = set()
        self.n_late = 0
        self.n_masked = 0
        self.n_unplanned = 0

    def reset(self):
        self.t = 0
        self.seen = set()

    def schedule(self, global_obs):
        n = self.num_datacenters
        planner = global_obs.get('planner') or {}
        ids = np.asarray(planner.get('batch_cloudlet_ids', [-1] * self.batch_size), dtype=np.int64)
        mask = global_obs.get('batch_cloudlet_offset_allowed')
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        grid = self.grid
        if not grid:
            from gym_cloudsimplus.envs.option_executor import offset_grid
            grid = offset_grid(int(os.environ.get("OFFSET_WAIT_CAP_STEPS", "72")))
        K = len(grid)
        if mask is not None and mask.ndim == 2 and mask.shape[1] != n * K:
            # the executor is on a different offset grid (e.g. the dyadic 12-value one without
            # OFFSET_GRID_DENSE=1): the action index would land on the wrong (site, κ); refuse
            raise RuntimeError(f"schedule_replay: executor grid has {mask.shape[1] // n} offsets, "
                               f"the schedule's grid has {K}; set OFFSET_GRID_DENSE=1 for a 0..W plan")
        out = []
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0:
                out.append(0)
                continue
            if jid not in self.plan:
                self.n_unplanned += 1
                out.append(0)
                continue
            site, start = self.plan[jid]
            kappa = start - self.t - self.START_LAG
            if kappa < 0:
                self.n_late += 1
                kappa = 0
            if kappa not in grid:
                kappa = max(g for g in grid if g <= kappa)       # every-step grid: exact
            a = site * K + grid.index(kappa)
            if kappa > 0 and mask is not None and j < mask.shape[0] and float(mask[j, a]) < 0.5:
                self.n_masked += 1
                a = site * K + 0
            out.append(a)
        self.t += 1
        return out


class CausalExpertGlobalScheduler(GlobalScheduler):
    """causal_expert (2026-09-06, the bridge between the offline exact ladder and RL): a rolling
    exact planner that at every step decides only the jobs that have ARRIVED and are not yet
    committed, using (i) those jobs, (ii) its own committed reservations (draw and occupancy
    per site and row, exactly what the every-step offset executor will run), and (iii) the
    arm's future green curve (CAUSAL_RUNG: truth, shrink_<lambda>, shuffle, anti, built from
    the wind files for the window like the ladder's rungs). It never sees a job before the
    simulator presents it. Each decision is the version-2 MILP of `ladder_planner` with the
    committed load as base, solved to proven optimality (envelope cuts), and emitted as
    (site, kappa) with kappa = start - t - 1 on the dense grid; the executor's legality mask
    restricts the candidate starts per site. Reads no answer of any other arm."""

    HANDLES_DEFER = True
    OPTION = True
    START_LAG = 1                      # executor: release at t + kappa, start at t + kappa + 1
    PLAN_LAG = 2                       # earliest start after sighting (ladder_planner.LAG)

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        import yaml
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        lad_dir = os.path.join(repo, "g1", "compressed_timecap_s2")
        if lad_dir not in sys.path:
            sys.path.insert(0, lad_dir)
        import ladder_run as lr                      # noqa: E402
        from ladder_planner import runtime_steps     # noqa: E402
        self._lr, self._runtime_steps = lr, runtime_steps
        cfg_path = os.environ.get("EVAL_CONFIG_PATH", "").strip()
        cell = os.environ.get("ORACLE_EXPERIMENT", "").strip()
        if not cfg_path or not cell:
            raise RuntimeError("causal_expert needs EVAL_CONFIG_PATH and ORACLE_EXPERIMENT")
        cfg = yaml.safe_load(open(cfg_path))
        self.blk = cfg[cell]
        self.sites = lr.sites_from_config(cfg, self.blk)
        if len(self.sites) != num_datacenters:
            raise RuntimeError(f"causal_expert: config has {len(self.sites)} sites, env {num_datacenters}")
        self.offset_rows = int(os.environ.get("ORACLE_OFFSET_ROWS", "0"))
        self.rung = os.environ.get("CAUSAL_RUNG", "truth")
        self.horizon_rows = int(os.environ.get("CAUSAL_HORIZON_ROWS", "1400"))
        self.wait_cap = int(os.environ.get("OFFSET_WAIT_CAP_STEPS", "72"))
        self.vm_mips = float(self.blk["datacenters"][0].get("vm_pe_mips", 40000))
        self.util = float(self.blk.get("cloudlet_cpu_utilization", 1.0))
        self.time_limit_s = float(os.environ.get("CAUSAL_TIME_LIMIT_S", "120"))
        truth, meta = lr.truth_curve(self.blk, self.offset_rows, self.horizon_rows)
        self.curve_w = lr.rung_curve(truth, self.rung, lr._mu_w(self.blk), seed_key=f"ladder:{self.offset_rows}")
        self.curve_signature = lr.curve_signature(self.curve_w)
        self.truth_signature = meta["signature"]
        self.reset()

    def reset(self):
        n, H = self.num_datacenters, self.horizon_rows
        self.t = 0
        self.plan = {}                                   # job id -> (site, start)
        self.committed_draw = np.zeros((n, H), dtype=np.int64)
        self.committed_occ = np.zeros((n, H + 1), dtype=np.int64)
        self.n_decisions = 0
        self.n_masked_fallback = 0
        self.n_unsolved = 0
        self.solve_wall_s = []
        self.solve_status = {}

    def _job_facts(self, planner, j):
        from ladder_planner import Job
        pes = int(float(np.asarray(planner.get("batch_cloudlet_pes"))[j]))
        mi = float(np.asarray(planner.get("batch_cloudlet_mi"))[j])
        ttd = planner.get("batch_cloudlet_time_to_deadline")
        present = planner.get("batch_cloudlet_deadline_present")
        r = self._runtime_steps(mi, self.vm_mips, self.util)
        has_dl = present is None or float(np.asarray(present)[j]) >= 0.5
        deadline = self.t + int(np.floor(float(np.asarray(ttd)[j]))) if (ttd is not None and has_dl) else 10 ** 9
        return Job(id=-1, arrival=self.t, runtime=r, pes=pes, deadline=deadline)

    def decide(self, t, new_jobs, masks, grid):
        """One proven-optimal MILP for the newly visible jobs on the committed load."""
        sched, res = causal_decide(t, new_jobs, masks, grid, self.sites, self.curve_w, self.committed_draw,
                                   self.committed_occ, start_lag=self.START_LAG, plan_lag=self.PLAN_LAG,
                                   time_limit_s=self.time_limit_s)
        self.solve_wall_s.append(float(res.get("wall_s", 0.0)))
        self.solve_status[t] = res.get("status")
        if not sched:
            self.n_unsolved += 1
        return sched, res

    def schedule(self, global_obs):
        n = self.num_datacenters
        planner = global_obs.get("planner") or {}
        ids = np.asarray(planner.get("batch_cloudlet_ids", [-1] * self.batch_size), dtype=np.int64)
        mask = global_obs.get("batch_cloudlet_offset_allowed")
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        grid = global_obs.get("offset_grid")
        if grid is None:
            from gym_cloudsimplus.envs.option_executor import offset_grid
            grid = offset_grid(self.wait_cap)
        grid = list(grid)
        K = len(grid)
        t = self.t
        new_jobs, masks, slot_of = {}, {}, {}
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0 or jid in self.plan:
                continue
            new_jobs[jid] = self._job_facts(planner, j)
            masks[jid] = None if mask is None or j >= mask.shape[0] else mask[j]
            slot_of[jid] = j
        if new_jobs:
            sched, _ = self.decide(t, new_jobs, masks, grid)
            for jid, (d, s) in sched.items():
                jb = new_jobs[jid]
                self.plan[jid] = (int(d), int(s))
                P = self.sites[d].job_power_mw(jb.pes)
                self.committed_draw[d, s:s + jb.runtime] += P
                self.committed_occ[d, s:s + jb.runtime + 1] += 1
                self.n_decisions += 1
        out = []
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0 or jid not in self.plan:
                out.append(0)
                if jid >= 0:
                    self.n_masked_fallback += 1           # unsolved this step: routed now (counted)
                continue
            d, s = self.plan[jid]
            kappa = s - t - self.START_LAG
            if kappa in grid:
                out.append(d * K + grid.index(kappa))
            else:
                out.append(d * K + 0)
                self.n_masked_fallback += 1
        self.t += 1
        return out

    def counters(self):
        return {"causal_decisions": self.n_decisions, "causal_unsolved": self.n_unsolved,
                "causal_fallback": self.n_masked_fallback, "causal_rung": self.rung,
                "causal_curve_signature": self.curve_signature, "causal_truth_signature": self.truth_signature,
                "causal_solve_wall_max_s": (max(self.solve_wall_s) if self.solve_wall_s else 0.0),
                "causal_solve_wall_sum_s": float(sum(self.solve_wall_s))}


def causal_decide(t, new_jobs, masks, grid, sites, curve_w, committed_draw, committed_occ,
                  start_lag=1, plan_lag=2, time_limit_s=120.0):
    """Pure core of the causal expert. new_jobs: {id: Job with arrival t}; masks: {id: the
    executor's legality row (n*K) or None}; grid: offset grid; curve_w: (n, H) W the arm's
    curve; committed_draw (n, H) mW and committed_occ (n, H+1) the reservations so far.
    Candidate start s = t + kappa + start_lag for every legal kappa >= 1 within the job's
    window. Returns ({id: (site, start)}, solver record); {} when not proven optimal."""
    from ladder_planner import build_instance, solve_milp, Job
    n, K = len(sites), len(grid)
    H = int(np.asarray(curve_w).shape[1])
    jobs, allowed = [], {}
    for jid, jb in new_jobs.items():
        job = Job(id=jid, arrival=t, runtime=jb.runtime, pes=jb.pes, deadline=jb.deadline)
        jobs.append(job)
        allowed[jid] = {}
        latest = min(job.latest, H - job.runtime - 1)
        row = masks.get(jid)
        for d in range(n):
            starts = []
            for i, kappa in enumerate(grid):
                if kappa < 1:
                    continue
                s = t + int(kappa) + start_lag
                if s < t + plan_lag or s > latest:
                    continue
                if row is not None and float(row[d * K + i]) < 0.5:
                    continue
                starts.append(s)
            allowed[jid][d] = starts
    inst = build_instance(jobs, list(sites), np.asarray(curve_w), base_draw_mw=committed_draw,
                          base_occ=committed_occ, starts_by_site=allowed)
    res = solve_milp(inst, time_limit_s=time_limit_s)
    if res.get("status") != "OPTIMAL":
        return {}, res
    return res["schedule"], res


class CoverArgmaxGlobalScheduler(GlobalScheduler):
    """cover_argmax (diagnostic for the F1-F3 reading, 2026-09-06): a zero-parameter arm that
    at each job's first sighting takes the legal (site, κ) with the largest `cand_green_cover`
    (ties: the smallest κ, then the lowest site). It reads exactly the F2/F3 interface key and
    nothing else, so its executed capture says whether the key itself carries the causal
    expert's value, separating "the interface is insufficient" from "the fit did not learn"."""

    HANDLES_DEFER = True
    OPTION = True

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        self.decided = set()
        self.n_decisions = 0
        self.n_no_cover = 0

    def reset(self):
        self.decided = set()

    def schedule(self, global_obs):
        n = self.num_datacenters
        planner = global_obs.get("planner") or {}
        ids = np.asarray(planner.get("batch_cloudlet_ids", [-1] * self.batch_size), dtype=np.int64)
        cover = global_obs.get("cand_green_cover")
        mask = global_obs.get("batch_cloudlet_offset_allowed")
        out = []
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0:
                out.append(0); continue
            a = cover_argmax_action(None if cover is None or j >= len(cover) else np.asarray(cover[j], dtype=np.float64),
                                    None if mask is None or j >= len(mask) else np.asarray(mask[j], dtype=np.float64), n)
            if jid not in self.decided:
                self.decided.add(jid); self.n_decisions += 1
                if cover is None:
                    self.n_no_cover += 1
            out.append(a)
        return out

    def counters(self):
        return {"cover_decisions": self.n_decisions, "cover_missing": self.n_no_cover}


def cover_argmax_action(cover_row, mask_row, num_dcs):
    """Pure: index a = d * K + i of the legal candidate with the largest cover; ties broken by
    the smallest offset index i, then the lowest site; 0 when nothing is legal or no cover."""
    if cover_row is None:
        return 0
    K = cover_row.shape[0] // num_dcs
    legal = np.ones_like(cover_row, dtype=bool) if mask_row is None else (mask_row >= 0.5)
    if not legal.any():
        return 0
    c = np.where(legal, cover_row, -np.inf)
    best = float(c.max())
    cands = np.where(c >= best - 1e-12)[0]
    return int(min(cands, key=lambda a: (a % K, a // K)))


class AlwaysHoldGlobalScheduler(GlobalScheduler):
    """Adversarial contract arm (OPTION_ACTION_DESIGN §5): HOLD at the greenest site now on
    every slot the hold mask allows, ROUTE_NOW there otherwise. Exercises the executor's
    fallback reservations and the deadline margin under a full hold backlog."""

    HANDLES_DEFER = True
    OPTION = True

    def schedule(self, global_obs):
        n = self.num_datacenters
        green = np.asarray(global_obs.get('dc_current_green_power_w', [0.0] * n), dtype=np.float64)
        planner = global_obs.get('planner') or {}
        ids = np.asarray(planner.get('batch_cloudlet_ids', [-1] * self.batch_size), dtype=np.int64)
        mask = global_obs.get('batch_cloudlet_hold_allowed')
        mask = None if mask is None else np.asarray(mask, dtype=np.float64)
        out = []
        for j in range(self.batch_size):
            jid = int(ids[j]) if j < ids.shape[0] else -1
            if jid < 0:
                out.append(0)
                continue
            allowed = None if mask is None or j >= mask.shape[0] else mask[j]
            if allowed is not None and float(allowed.max()) >= 0.5:
                order = np.argsort(-green)
                site = int(next(d for d in order if allowed[d] >= 0.5))
                out.append(n + site)
            else:
                out.append(int(np.argmax(green)))
        return out


class OptionBCGlobalScheduler(GlobalScheduler):
    """Executed behaviour-cloned option policy (OPTION_ACTION_DESIGN §6 gate 4, criterion 2).

    Loads the gate-4 fit (OPTION_BC_MODEL = directory holding model.pt) and the option block
    it was built from (OPTION_BC_CONFIG, OPTION_BC_BLOCK), runs the module recurrently on the
    raw env observation with the legality mask applied, and acts by argmax over the 2n
    choices (deterministic decode, recorded in fit.json). Illegal choices cannot occur at
    the logit level; the env's translation counts any that would.
    """

    HANDLES_DEFER = True
    OPTION = True

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        from src.baselines.option_bc_module import load_fitted_module
        model_dir = os.environ.get("OPTION_BC_MODEL", "").strip()
        cfg = os.environ.get("OPTION_BC_CONFIG", "").strip()
        block = os.environ.get("OPTION_BC_BLOCK", "").strip()
        if not (model_dir and cfg and block):
            raise RuntimeError("option_bc needs OPTION_BC_MODEL, OPTION_BC_CONFIG and OPTION_BC_BLOCK")
        self.module, _obs_space, act_space = load_fitted_module(model_dir, cfg, block)
        nvec = [int(x) for x in act_space.nvec]
        if len(nvec) != batch_size or nvec[0] % num_datacenters != 0 or nvec[0] < 2 * num_datacenters:
            raise RuntimeError(f"option_bc module space {nvec[:1]}x{len(nvec)} does not match the env "
                               f"(a multiple of {num_datacenters} choices x {batch_size} slots)")
        self.n_choices = nvec[0]
        self._state = None
        self.n_steps = 0

    def reset(self):
        self._state = None
        self.n_steps = 0

    def schedule(self, global_obs):
        import torch
        raw = global_obs.get("raw_global_obs")
        if raw is None:
            raise RuntimeError("option_bc needs the raw global observation (evaluate forwards it as raw_global_obs)")
        obs = {k: torch.as_tensor(np.asarray(v)[None, ...]) for k, v in raw.items()}
        batch = {"obs": {"observation": obs, "action_mask": torch.ones(1, self.batch_size)}}
        if self._state is None:
            self._state = self.module.get_initial_state()
        batch["state_in"] = {k: torch.as_tensor(v) for k, v in self._state.items()} if isinstance(self._state, dict) \
            else {"gtrxl_mem": torch.as_tensor(self._state)}
        with torch.no_grad():
            out = self.module.forward_inference(batch)
        logits = out["action_dist_inputs"].reshape(-1, self.batch_size, self.n_choices)[-1]
        so = out.get("state_out")
        if so is not None:
            self._state = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 4 else v) for k, v in so.items()} \
                if isinstance(so, dict) else (so[0] if so.dim() == 4 else so)
        self.n_steps += 1
        return [int(a) for a in logits.argmax(-1).tolist()]


GLOBAL_SCHEDULERS = {
    'curve_planner_off': OraclePlannerOffsetGlobalScheduler,
    'perturbed_oracle_planner_off': PerturbedOraclePlannerOffsetGlobalScheduler,
    'persistence_planner_off': PersistencePlannerOffsetGlobalScheduler,
    'climatology_planner_off': ClimatologyPlannerOffsetGlobalScheduler,
    'reactive_wait_planner_off': ReactiveWaitPlannerOffsetGlobalScheduler,
    'fixed_off': FixedOffsetGlobalScheduler,
    'schedule_replay': ScheduleReplayGlobalScheduler,
    'causal_expert': CausalExpertGlobalScheduler,
    'cover_argmax': CoverArgmaxGlobalScheduler,
    'option_bc': OptionBCGlobalScheduler,
    'curve_planner_opt': OraclePlannerOptionGlobalScheduler,
    'perturbed_oracle_planner_opt': PerturbedOraclePlannerOptionGlobalScheduler,
    'persistence_planner_opt': PersistencePlannerOptionGlobalScheduler,
    'climatology_planner_opt': ClimatologyPlannerOptionGlobalScheduler,
    'reactive_wait_planner_opt': ReactiveWaitPlannerOptionGlobalScheduler,
    'always_hold': AlwaysHoldGlobalScheduler,
    'green_forecast_queue_balanced': GreenForecastQueueBalancedGlobalScheduler,
    'random': RandomGlobalScheduler,
    'round_robin': RoundRobinGlobalScheduler,
    'min_queue': MinQueueGlobalScheduler,
    'green_aware': GreenAwareGlobalScheduler,
    'green_forecast': GreenForecastAwareGlobalScheduler,
    'green_forecast_capacity': GreenForecastCapacityGlobalScheduler,
    'always_defer': AlwaysDeferGlobalScheduler,
    'curve_planner': CurveInformedPlannerGlobalScheduler,
    'oracle144_planner': HorizonLimitedOraclePlannerGlobalScheduler,
    'perturbed_oracle_planner': PerturbedOraclePlannerGlobalScheduler,
    'load_smoothing': LoadSmoothingGlobalScheduler,
    'reservation_edf': ReservationEDFGlobalScheduler,
    'persistence_planner': PersistencePlannerGlobalScheduler,
    'climatology_planner': ClimatologyPlannerGlobalScheduler,
    'reactive_wait_planner': ReactiveWaitPlannerGlobalScheduler,
    'nowait_planner': NoWaitPlannerGlobalScheduler,
    'green_queue_balanced': GreenQueueBalancedGlobalScheduler,
    'min_brown_power': MinBrownPowerGlobalScheduler,
    'weighted_score': WeightedScoreGlobalScheduler,
    'ga': GeneticAlgorithmGlobalScheduler,
    'pso': ParticleSwarmGlobalScheduler,
    'rllib': RLlibGlobalScheduler,  # For Multi-DC (RLlib/Ray) - Old API
    'rllib_new_api': RLlibNewAPIGlobalScheduler,  # For New API Stack (RLModule)
}
