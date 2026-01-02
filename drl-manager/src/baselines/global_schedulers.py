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
        # Assign Cloudlet to the DC with highest Green Energy
        best_dc = int(np.argmax(green_ratios))
        return [best_dc] * self.batch_size


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
            best_dc = int(np.argmin(brown))
        else:
            best_dc = int(np.argmax(green_ratio))

        return [best_dc] * self.batch_size


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
    """

    def __init__(
        self,
        num_datacenters: int,
        batch_size: int,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.08,
        tournament_k: int = 3,
        w_queue: float = 1.0,
        w_imbalance: float = 0.5,
        w_brown: float = 0.3,
        w_green: float = 0.3,
        w_capacity: float = 2.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(num_datacenters, batch_size)
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = max(2, int(tournament_k))
        self.w_queue = float(w_queue)
        self.w_imbalance = float(w_imbalance)
        self.w_brown = float(w_brown)
        self.w_green = float(w_green)
        self.w_capacity = float(w_capacity)
        self.rng = np.random.default_rng(rng_seed)

    def _fitness(
        self,
        assignment: np.ndarray,
        queue: np.ndarray,
        green_ratio: np.ndarray,
        brown_power: np.ndarray,
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

        brown_term = float(np.mean(brown_power[assignment])) if brown_power.size else 0.0

        # Capacity penalty: if available_pes is low compared to demand, penalize.
        # (Soft proxy; local scheduler can still queue/noassign.)
        cap_pen = 0.0
        if avail_pes.size and batch_pes.size:
            demands = batch_pes[: assignment.size].astype(np.float32)
            shortfall = np.maximum(0.0, demands - avail_pes[assignment].astype(np.float32))
            cap_pen = float(np.mean(shortfall))

        return (
            self.w_queue * queue_pressure
            + self.w_imbalance * imbalance
            + self.w_brown * brown_term
            - self.w_green * green_term
            + self.w_capacity * cap_pen
        )

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idxs = self.rng.integers(0, fitnesses.size, size=self.tournament_k)
        best = idxs[np.argmin(fitnesses[idxs])]
        return int(best)

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        b = self.batch_size

        queue = _as_np_1d(global_obs.get("dc_queue_sizes"), n, fill=0.0, dtype=np.float32)
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), n, fill=0.5, dtype=np.float32)
        avail_pes = _as_np_1d(global_obs.get("dc_available_pes"), n, fill=0.0, dtype=np.float32)
        batch_pes = np.ravel(np.array(global_obs.get("batch_cloudlet_pes", []), dtype=np.float32))
        if batch_pes.size < b:
            # if missing, treat all as 1 PE so capacity penalty becomes mild
            batch_pes = np.ones(b, dtype=np.float32)

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

        # Seed 0: greedy min-queue (with simulated increments)
        tmp_q = queue.copy()
        seed0 = np.empty(b, dtype=np.int32)
        for i in range(b):
            dc = int(np.argmin(tmp_q))
            seed0[i] = dc
            tmp_q[dc] += 1.0
        pop[0] = seed0

        # Seed 1: greedy weighted score (green vs queue)
        tmp_q = queue.copy()
        seed1 = np.empty(b, dtype=np.int32)
        g_norm = _normalize_01(green_ratio)
        for i in range(b):
            q_norm = _normalize_01(tmp_q)
            score = 0.6 * g_norm + 0.4 * (1.0 - q_norm)
            dc = int(np.argmax(score))
            seed1[i] = dc
            tmp_q[dc] += 1.0
        if self.population_size > 1:
            pop[1] = seed1

        # Remaining: random
        for i in range(2, self.population_size):
            pop[i] = self.rng.integers(0, n, size=b, dtype=np.int32)

        fitnesses = np.array(
            [self._fitness(ind, queue, green_ratio, brown_power, avail_pes, batch_pes) for ind in pop],
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
                [self._fitness(ind, queue, green_ratio, brown_power, avail_pes, batch_pes) for ind in pop],
                dtype=np.float32,
            )

        best_idx = int(np.argmin(fitnesses))
        return pop[best_idx].tolist()


class ParticleSwarmGlobalScheduler(GlobalScheduler):
    """
    Lightweight PSO baseline for batch routing (discrete via rounding).

    Particle position: real-valued vector (len=batch_size) mapped to DC ids by rounding+clamp.
    Objective: same as GA (queue pressure + imbalance + brown power - green + capacity penalty).
    """

    def __init__(
        self,
        num_datacenters: int,
        batch_size: int,
        swarm_size: int = 18,
        iterations: int = 12,
        inertia: float = 0.6,
        c1: float = 1.4,
        c2: float = 1.4,
        w_queue: float = 1.0,
        w_imbalance: float = 0.5,
        w_brown: float = 0.3,
        w_green: float = 0.3,
        w_capacity: float = 2.0,
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
        self.w_brown = float(w_brown)
        self.w_green = float(w_green)
        self.w_capacity = float(w_capacity)
        self.rng = np.random.default_rng(rng_seed)

    def _fitness(
        self,
        assignment: np.ndarray,
        queue: np.ndarray,
        green_ratio: np.ndarray,
        brown_power: np.ndarray,
        avail_pes: np.ndarray,
        batch_pes: np.ndarray,
    ) -> float:
        # Reuse GA fitness structure
        n = self.num_datacenters
        add = _bincount_assignments(assignment, n).astype(np.float32)
        q_pred = queue.astype(np.float32) + add
        queue_pressure = float(np.mean(q_pred))
        imbalance = float(np.std(q_pred))
        green_term = float(np.mean(green_ratio[assignment])) if green_ratio.size else 0.0
        brown_term = float(np.mean(brown_power[assignment])) if brown_power.size else 0.0
        cap_pen = 0.0
        if avail_pes.size and batch_pes.size:
            demands = batch_pes[: assignment.size].astype(np.float32)
            shortfall = np.maximum(0.0, demands - avail_pes[assignment].astype(np.float32))
            cap_pen = float(np.mean(shortfall))
        return (
            self.w_queue * queue_pressure
            + self.w_imbalance * imbalance
            + self.w_brown * brown_term
            - self.w_green * green_term
            + self.w_capacity * cap_pen
        )

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        n = self.num_datacenters
        b = self.batch_size

        queue = _as_np_1d(global_obs.get("dc_queue_sizes"), n, fill=0.0, dtype=np.float32)
        green_ratio = _as_np_1d(global_obs.get("dc_green_ratio"), n, fill=0.5, dtype=np.float32)
        avail_pes = _as_np_1d(global_obs.get("dc_available_pes"), n, fill=0.0, dtype=np.float32)
        batch_pes = np.ravel(np.array(global_obs.get("batch_cloudlet_pes", []), dtype=np.float32))
        if batch_pes.size < b:
            batch_pes = np.ones(b, dtype=np.float32)

        current_power = _as_np_1d(global_obs.get("dc_current_power_w"), n, fill=np.nan, dtype=np.float32)
        if np.isfinite(current_power).any():
            finite = current_power[np.isfinite(current_power)]
            max_finite = float(np.max(finite)) if finite.size else 0.0
            cur = np.where(np.isfinite(current_power), current_power, max_finite)
            brown_power = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
        else:
            brown_power = np.zeros(n, dtype=np.float32)

        # init swarm near heuristic solutions + random
        # positions in [0, n-1]
        pos = self.rng.random((self.swarm_size, b), dtype=np.float32) * float(max(n - 1, 1))
        vel = (self.rng.random((self.swarm_size, b), dtype=np.float32) - 0.5) * 0.5

        # seed particle 0 with greedy min-queue
        tmp_q = queue.copy()
        seed = np.empty(b, dtype=np.float32)
        for i in range(b):
            dc = int(np.argmin(tmp_q))
            seed[i] = float(dc)
            tmp_q[dc] += 1.0
        pos[0] = seed

        def to_assign(p: np.ndarray) -> np.ndarray:
            a = np.rint(p).astype(np.int32)
            return np.clip(a, 0, n - 1)

        pbest_pos = pos.copy()
        pbest_fit = np.array(
            [self._fitness(to_assign(p), queue, green_ratio, brown_power, avail_pes, batch_pes) for p in pos],
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
                [self._fitness(to_assign(p), queue, green_ratio, brown_power, avail_pes, batch_pes) for p in pos],
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


def load_rllib_algorithm(checkpoint_path: str):
    """
    load RLlib checkpoint and return Algorithm instance

    Args:
        checkpoint_path: checkpoint path

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

    # register environment
    from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv

    def env_creator(cfg):
        env = HierarchicalMultiDCParallelEnv(cfg)
        return ParallelPettingZooEnv(env)

    tune.register_env('multidc_env', env_creator)

    # load checkpoint (using absolute file:// URI, to avoid pyarrow error for relative paths)
    checkpoint_uri = Path(checkpoint_path).resolve().as_uri()
    algo = Algorithm.from_checkpoint(checkpoint_uri)
    return algo


# === Register all global schedulers ===
GLOBAL_SCHEDULERS = {
    'random': RandomGlobalScheduler,
    'round_robin': RoundRobinGlobalScheduler,
    'min_queue': MinQueueGlobalScheduler,
    'green_aware': GreenAwareGlobalScheduler,
    'green_queue_balanced': GreenQueueBalancedGlobalScheduler,
    'min_brown_power': MinBrownPowerGlobalScheduler,
    'weighted_score': WeightedScoreGlobalScheduler,
    'ga': GeneticAlgorithmGlobalScheduler,
    'pso': ParticleSwarmGlobalScheduler,
    'rllib': RLlibGlobalScheduler,  # For Multi-DC (RLlib/Ray)
}
