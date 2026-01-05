import numpy as np
from typing import Dict, Any, Optional
from .base import LocalScheduler

# Type hint for RLlib Algorithm (avoid import at module level)
AlgorithmType = Any


def _valid_action_indices(action_mask: np.ndarray) -> np.ndarray:
    """Return valid action indices (including 0 for NoAssign if allowed)."""
    if action_mask is None:
        return np.array([0], dtype=np.int32)
    mask = np.asarray(action_mask).astype(bool)
    idx = np.where(mask)[0].astype(np.int32)
    return idx if idx.size > 0 else np.array([0], dtype=np.int32)


class RandomLocalScheduler(LocalScheduler):
    """Random Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        valid_actions = _valid_action_indices(action_mask)
        if len(valid_actions) > 0:
            return int(np.random.choice(valid_actions))
        return 0  # NoAssign


class FirstFitLocalScheduler(LocalScheduler):
    """First Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        # Start from VM 1
        for i in range(1, len(action_mask)):
            if action_mask[i]:
                return i
        return 0  # NoAssign


class BestFitLocalScheduler(LocalScheduler):
    """Best Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_available_pes = local_obs.get('vm_available_pes', [])
        next_cloudlet_pes = local_obs.get('next_cloudlet_pes', 1)

        best_vm = 0
        min_waste = float('inf')

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_available_pes):
                available = vm_available_pes[i-1]
                waste = available - next_cloudlet_pes
                if 0 <= waste < min_waste:
                    min_waste = waste
                    best_vm = i

        return best_vm


class WorstFitLocalScheduler(LocalScheduler):
    """Worst Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_available_pes = local_obs.get('vm_available_pes', [])
        next_cloudlet_pes = local_obs.get('next_cloudlet_pes', 1)

        best_vm = 0
        max_remaining = -1

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_available_pes):
                available = vm_available_pes[i-1]
                remaining = available - next_cloudlet_pes
                if remaining >= 0 and remaining > max_remaining:
                    max_remaining = remaining
                    best_vm = i

        return best_vm


class RoundRobinLocalScheduler(LocalScheduler):
    """Round Robin Local Scheduler"""

    def __init__(self, num_vms: int):
        super().__init__(num_vms)
        self.current_vm = 1

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        for _ in range(self.num_vms):
            if self.current_vm < len(action_mask) and action_mask[self.current_vm]:
                selected = self.current_vm
                self.current_vm = (self.current_vm % self.num_vms) + 1
                return selected
            self.current_vm = (self.current_vm % self.num_vms) + 1
        return 0  # NoAssign

    def reset(self):
        self.current_vm = 1


class MinLoadLocalScheduler(LocalScheduler):
    """Greedy Min Load Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_loads = local_obs.get('vm_loads', [])

        best_vm = 0
        min_load = float('inf')

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_loads):
                load = vm_loads[i-1]
                if load < min_load:
                    min_load = load
                    best_vm = i

        return best_vm


class PackingAwareLocalScheduler(LocalScheduler):
    """
    Packing-aware local heuristic (green-friendly proxy):

    Prefer to *pack* cloudlets onto already-loaded VMs/hosts (helps consolidation),
    but avoid overload and avoid large PE waste.

    Cost per VM (lower is better):
      + a * vm_load
      + b * waste_pes
      + c * overload_penalty
    """

    def __init__(self, num_vms: int, a: float = 1.0, b: float = 0.5, c: float = 5.0):
        super().__init__(num_vms)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        valid = _valid_action_indices(action_mask)
        # If only NoAssign is valid, return it.
        if valid.size == 1 and int(valid[0]) == 0:
            return 0

        vm_loads = np.array(local_obs.get("vm_loads", []), dtype=np.float32).ravel()
        vm_avail = np.array(local_obs.get("vm_available_pes", []), dtype=np.float32).ravel()
        demand = float(local_obs.get("next_cloudlet_pes", 0) or 0.0)

        best_action = 0
        best_cost = float("inf")
        for a in valid:
            a = int(a)
            if a == 0:
                # NoAssign: allow but discourage if there is any feasible VM action
                continue
            idx = a - 1
            if idx >= vm_avail.size:
                continue
            avail = float(vm_avail[idx])
            if avail <= 0:
                continue
            waste = max(0.0, avail - demand)
            overload = max(0.0, demand - avail)  # should be 0 when mask is correct
            load = float(vm_loads[idx]) if idx < vm_loads.size else 0.0
            cost = self.a * load + self.b * waste + self.c * overload
            if cost < best_cost:
                best_cost = cost
                best_action = a

        return int(best_action) if best_action != 0 else int(valid[0])


class GeneticAlgorithmLocalScheduler(LocalScheduler):
    """
    GA baseline for choosing a single VM action under action_mask.

    This is intentionally lightweight: population is a set of candidate actions,
    evolved using tournament selection and mutation. Fitness uses a multi-criteria
    local cost (pack vs waste vs overload).
    """

    def __init__(
        self,
        num_vms: int,
        population_size: int = 18,
        generations: int = 10,
        mutation_rate: float = 0.25,
        tournament_k: int = 3,
        a: float = 1.0,
        b: float = 0.5,
        c: float = 5.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(num_vms)
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = max(2, int(tournament_k))
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.rng = np.random.default_rng(rng_seed)

    def _cost(self, action: int, vm_loads: np.ndarray, vm_avail: np.ndarray, demand: float) -> float:
        if action == 0:
            # Penalize NoAssign if other options exist; caller will handle feasibility.
            return 1e6
        idx = action - 1
        if idx < 0 or idx >= vm_avail.size:
            return 1e6
        avail = float(vm_avail[idx])
        if avail <= 0:
            return 1e6
        waste = max(0.0, avail - demand)
        overload = max(0.0, demand - avail)
        load = float(vm_loads[idx]) if idx < vm_loads.size else 0.0
        return self.a * load + self.b * waste + self.c * overload

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idxs = self.rng.integers(0, fitnesses.size, size=self.tournament_k)
        best = idxs[np.argmin(fitnesses[idxs])]
        return int(best)

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        valid = _valid_action_indices(action_mask)
        if valid.size == 1:
            return int(valid[0])

        # Exclude 0 from normal population if possible (keep as fallback)
        valid_nonzero = valid[valid != 0]
        if valid_nonzero.size == 0:
            return 0

        vm_loads = np.array(local_obs.get("vm_loads", []), dtype=np.float32).ravel()
        vm_avail = np.array(local_obs.get("vm_available_pes", []), dtype=np.float32).ravel()
        demand = float(local_obs.get("next_cloudlet_pes", 0) or 0.0)

        # init population from heuristics + random valid actions
        pop = np.empty(self.population_size, dtype=np.int32)
        # seed with BestFit (min waste among feasible)
        bestfit = 0
        min_waste = float("inf")
        for a in valid_nonzero:
            idx = int(a) - 1
            if idx < vm_avail.size:
                waste = float(vm_avail[idx]) - demand
                if 0.0 <= waste < min_waste:
                    min_waste = waste
                    bestfit = int(a)
        pop[0] = bestfit if bestfit != 0 else int(valid_nonzero[0])

        # seed with MinLoad
        minload = int(valid_nonzero[0])
        best_load = float("inf")
        for a in valid_nonzero:
            idx = int(a) - 1
            load = float(vm_loads[idx]) if idx < vm_loads.size else 0.0
            if load < best_load:
                best_load = load
                minload = int(a)
        if self.population_size > 1:
            pop[1] = minload

        # rest random
        for i in range(2, self.population_size):
            pop[i] = int(self.rng.choice(valid_nonzero))

        fitnesses = np.array([self._cost(int(a), vm_loads, vm_avail, demand) for a in pop], dtype=np.float32)

        for _ in range(self.generations):
            new_pop = np.empty_like(pop)
            elite_idx = int(np.argmin(fitnesses))
            new_pop[0] = pop[elite_idx]

            for i in range(1, self.population_size):
                p1 = int(pop[self._tournament(fitnesses)])
                p2 = int(pop[self._tournament(fitnesses)])
                # crossover in single-gene space: pick one parent
                child = p1 if self.rng.random() < 0.5 else p2
                if self.rng.random() < self.mutation_rate:
                    child = int(self.rng.choice(valid_nonzero))
                new_pop[i] = child

            pop = new_pop
            fitnesses = np.array([self._cost(int(a), vm_loads, vm_avail, demand) for a in pop], dtype=np.float32)

        best = int(pop[int(np.argmin(fitnesses))])
        return best if best in set(valid.tolist()) else int(valid[0])


class ParticleSwarmLocalScheduler(LocalScheduler):
    """
    PSO baseline for choosing a single VM action under action_mask.

    Uses a 1D continuous position mapped to a discrete action via rounding and
    snapping to the nearest valid action.
    """

    def __init__(
        self,
        num_vms: int,
        swarm_size: int = 16,
        iterations: int = 12,
        inertia: float = 0.6,
        c1: float = 1.4,
        c2: float = 1.4,
        a: float = 1.0,
        b: float = 0.5,
        c: float = 5.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(num_vms)
        self.swarm_size = max(4, int(swarm_size))
        self.iterations = max(1, int(iterations))
        self.inertia = float(inertia)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.rng = np.random.default_rng(rng_seed)

    def _cost(self, action: int, vm_loads: np.ndarray, vm_avail: np.ndarray, demand: float) -> float:
        if action == 0:
            return 1e6
        idx = action - 1
        if idx < 0 or idx >= vm_avail.size:
            return 1e6
        avail = float(vm_avail[idx])
        if avail <= 0:
            return 1e6
        waste = max(0.0, avail - demand)
        overload = max(0.0, demand - avail)
        load = float(vm_loads[idx]) if idx < vm_loads.size else 0.0
        return self.a * load + self.b * waste + self.c * overload

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        valid = _valid_action_indices(action_mask)
        if valid.size == 1:
            return int(valid[0])

        valid_nonzero = valid[valid != 0]
        if valid_nonzero.size == 0:
            return 0

        vm_loads = np.array(local_obs.get("vm_loads", []), dtype=np.float32).ravel()
        vm_avail = np.array(local_obs.get("vm_available_pes", []), dtype=np.float32).ravel()
        demand = float(local_obs.get("next_cloudlet_pes", 0) or 0.0)

        # map continuous -> discrete action by rounding then snapping to nearest valid_nonzero
        valid_actions = valid_nonzero.astype(np.int32)

        def snap(x: float) -> int:
            a = int(np.rint(x))
            # clamp to plausible range then snap
            a = max(int(valid_actions.min()), min(int(valid_actions.max()), a))
            # nearest valid
            nearest = int(valid_actions[np.argmin(np.abs(valid_actions - a))])
            return nearest

        # init swarm around random valid actions
        pos = self.rng.choice(valid_actions, size=self.swarm_size).astype(np.float32)
        vel = (self.rng.random(self.swarm_size, dtype=np.float32) - 0.5) * 0.5

        pbest_pos = pos.copy()
        pbest_fit = np.array([self._cost(snap(float(p)), vm_loads, vm_avail, demand) for p in pos], dtype=np.float32)
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = float(pbest_pos[gbest_idx])
        gbest_fit = float(pbest_fit[gbest_idx])

        for _ in range(self.iterations):
            r1 = self.rng.random(self.swarm_size, dtype=np.float32)
            r2 = self.rng.random(self.swarm_size, dtype=np.float32)
            vel = self.inertia * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            pos = pos + vel

            fits = np.array([self._cost(snap(float(p)), vm_loads, vm_avail, demand) for p in pos], dtype=np.float32)
            improved = fits < pbest_fit
            if improved.any():
                pbest_fit = np.where(improved, fits, pbest_fit)
                pbest_pos = np.where(improved, pos, pbest_pos)
                new_gbest_idx = int(np.argmin(pbest_fit))
                new_gbest_fit = float(pbest_fit[new_gbest_idx])
                if new_gbest_fit < gbest_fit:
                    gbest_fit = new_gbest_fit
                    gbest_pos = float(pbest_pos[new_gbest_idx])

        best_action = snap(gbest_pos)
        return best_action if best_action in set(valid.tolist()) else int(valid[0])


class RLlibLocalScheduler(LocalScheduler):
    """
    RLlib-based Local Scheduler (for Multi-DC training with Ray)

    Uses shared Algorithm instance for all DCs.
    Each DC uses its own policy (local_policy_{dc_id}).
    """

    def __init__(self, num_vms: int, algo, dc_id: int, env=None, policy_id: Optional[str] = None,
                 use_parameter_sharing: bool = False, num_datacenters: int = 10,
                 max_hosts: int = 16, max_vms: int = 224):
        """
        Args:
            num_vms: Max number of VMs across all datacenters (for base class)
            algo: Shared RLlib Algorithm instance
            dc_id: Datacenter ID (for policy selection)
            env: HierarchicalMultiDCEnv instance (used to trim obs/mask to DC-specific sizes)
            use_parameter_sharing: Whether using shared_local_policy (parameter sharing mode)
            num_datacenters: Number of datacenters (for dc_id_onehot)
            max_hosts: Max hosts per DC (for padding)
            max_vms: Max VMs per DC (for padding)
        """
        super().__init__(num_vms)
        self.algo = algo
        self.dc_id = dc_id
        self.env = env
        self.use_parameter_sharing = use_parameter_sharing
        self.num_datacenters = num_datacenters
        self.max_hosts = max_hosts
        self.max_vms = max_vms
        # If a specific policy_id is given (e.g. for parameter sharing with
        # "shared_local_policy"), use it; otherwise default to per-DC policy.
        self.policy_id = policy_id or f"local_policy_{dc_id}"

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        Use RLlib model to select VM

        RLlib handles action mask through the 'action_mask' key in the observation.

        For parameter sharing mode (shared_local_policy):
        - Use full padded observations (max_hosts, max_vms)
        - Add dc_id_onehot and valid_vm_mask

        For independent policy mode:
        - Trim observations to each DC's actual host/vm count
        """
        if self.use_parameter_sharing:
            # Parameter sharing mode: use full padded observations
            # Mirror the PettingZoo wrapper's unified observation format
            dc_vm_count = self.env._get_dc_vm_count(self.dc_id) if self.env else self.max_vms

            # Build valid_vm_mask (1 for real VMs, 0 for padding)
            valid_vm_mask = np.zeros(self.max_vms, dtype=np.float32)
            valid_vm_mask[:dc_vm_count] = 1.0

            # DC ID one-hot
            dc_id_onehot = np.zeros(self.num_datacenters, dtype=np.float32)
            dc_id_onehot[self.dc_id] = 1.0

            # Use full padded observations (already padded by HierarchicalMultiDCEnv)
            unified_obs = {
                "host_loads": local_obs["host_loads"],
                "host_ram_usage": local_obs["host_ram_usage"],
                "vm_loads": local_obs["vm_loads"],
                "vm_types": local_obs["vm_types"],
                "vm_available_pes": local_obs["vm_available_pes"],
                "waiting_cloudlets": local_obs["waiting_cloudlets"],
                "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
                "dc_id_onehot": dc_id_onehot,
                "valid_vm_mask": valid_vm_mask,
            }

            # Full action mask (max_vms + 1)
            full_mask = action_mask.astype(np.float32)

            obs_with_mask = {
                "observation": unified_obs,
                "action_mask": full_mask,
            }
        else:
            # Non-parameter-sharing mode: trim to DC-specific sizes
            if self.env is not None:
                try:
                    dc_vm_count = self.env._get_dc_vm_count(self.dc_id)
                    dc_host_count = self.env._get_dc_host_count(self.dc_id)

                    trimmed_obs = {
                        "host_loads": local_obs["host_loads"][:dc_host_count],
                        "host_ram_usage": local_obs["host_ram_usage"][:dc_host_count],
                        "vm_loads": local_obs["vm_loads"][:dc_vm_count],
                        "vm_types": local_obs["vm_types"][:dc_vm_count],
                        "vm_available_pes": local_obs["vm_available_pes"][:dc_vm_count],
                        "waiting_cloudlets": local_obs["waiting_cloudlets"],
                        "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
                    }
                    trimmed_mask = action_mask[: dc_vm_count + 1].astype(np.float32)
                except Exception:
                    # Fallback to original obs/mask on any error
                    trimmed_obs = local_obs
                    trimmed_mask = action_mask.astype(np.float32)
            else:
                trimmed_obs = local_obs
                trimmed_mask = action_mask.astype(np.float32)

            obs_with_mask = {
                "observation": trimmed_obs,
                "action_mask": trimmed_mask,
            }

        # select policy_id：
        # - If local worker contains shared_local_policy，and current policy_id is not in the policy_map，
        #   Which means paremeterSharing is enabled, fallback to shared_local_policy, avoid accessing non-existent local_policy_*。
        policy_id = self.policy_id
        worker_mgr = getattr(self.algo, "workers", None)
        if worker_mgr is not None:
            try:
                local_worker = self.algo.workers.local_worker()
                policy_ids = set(local_worker.policy_map.keys())
                if "shared_local_policy" in policy_ids and policy_id not in policy_ids:
                    policy_id = "shared_local_policy"
            except Exception:
                # if cannot read policy_map, continue using original policy_id, let RLlib decide whether to error
                pass

        action = self.algo.compute_single_action(
            obs_with_mask,
            policy_id=policy_id,
            explore=False
        )

        return int(action)


def create_rllib_schedulers(algo, env, num_dcs: int, batch_size: int, num_vms: int,
                            max_hosts: int = 16):
    """
    create RLlib-based Global and Local schedulers

    Args:
        algo: loaded RLlib Algorithm instance
        env: HierarchicalMultiDCEnv instance (used to trim local observations to each DC's actual size)
        num_dcs: number of datacenters
        batch_size: Global routing batch size
        num_vms: maximum number of VMs per DC
        max_hosts: maximum number of hosts per DC (for parameter sharing)

    Returns:
        (global_scheduler, local_schedulers_dict)
    """
    from .global_schedulers import RLlibGlobalScheduler

    global_scheduler = RLlibGlobalScheduler(num_dcs, batch_size, algo)

    # Inspect the Algorithm's local worker policy map to detect whether
    # parameter sharing was used (i.e., presence of "shared_local_policy").
    worker = getattr(algo, "workers", None)
    policy_ids = set()
    if worker is not None:
        try:
            local_worker = algo.workers.local_worker()
            policy_ids = set(local_worker.policy_map.keys())
        except Exception:
            policy_ids = set()

    has_shared_local = "shared_local_policy" in policy_ids

    local_schedulers = {}
    for dc_id in range(num_dcs):
        if has_shared_local:
            # All DCs share the same local policy trained with parameter sharing.
            pid = "shared_local_policy"
        else:
            pid = f"local_policy_{dc_id}"
        local_schedulers[dc_id] = RLlibLocalScheduler(
            num_vms, algo, dc_id, env=env, policy_id=pid,
            use_parameter_sharing=has_shared_local,
            num_datacenters=num_dcs,
            max_hosts=max_hosts,
            max_vms=num_vms,
        )

    return global_scheduler, local_schedulers


class RLlibNewAPILocalScheduler(LocalScheduler):
    """
    RLlib-based Local Scheduler for New API Stack (RLModule).

    Uses RLModule directly for inference instead of compute_single_action.
    This is required for models trained with enable_rl_module_and_learner=True.
    """

    def __init__(self, num_vms: int, algo, dc_id: int, env=None, policy_id: Optional[str] = None,
                 use_parameter_sharing: bool = False, num_datacenters: int = 10,
                 max_hosts: int = 16, max_vms: int = 224):
        """
        Args:
            num_vms: Max number of VMs across all datacenters
            algo: RLlib Algorithm instance (New API Stack)
            dc_id: Datacenter ID
            env: HierarchicalMultiDCEnv instance
            policy_id: Policy ID to use
            use_parameter_sharing: Whether using shared_local_policy
            num_datacenters: Number of datacenters
            max_hosts: Max hosts per DC
            max_vms: Max VMs per DC
        """
        super().__init__(num_vms)
        self.algo = algo
        self.dc_id = dc_id
        self.env = env
        self.use_parameter_sharing = use_parameter_sharing
        self.num_datacenters = num_datacenters
        self.max_hosts = max_hosts
        self.max_vms = max_vms
        self.policy_id = policy_id or "shared_local_policy"

        # Get the RLModule
        self._rl_module = self._get_rl_module()

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

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        Use RLModule to select VM.
        """
        import torch

        if self._rl_module is None:
            raise RuntimeError("RLModule not available. Make sure algorithm uses New API Stack.")

        # Build observation with action mask (parameter sharing mode)
        dc_vm_count = self.env._get_dc_vm_count(self.dc_id) if self.env else self.max_vms

        # Build valid_vm_mask (1 for real VMs, 0 for padding)
        valid_vm_mask = np.zeros(self.max_vms, dtype=np.float32)
        valid_vm_mask[:dc_vm_count] = 1.0

        # DC ID one-hot
        dc_id_onehot = np.zeros(self.num_datacenters, dtype=np.float32)
        dc_id_onehot[self.dc_id] = 1.0

        unified_obs = {
            "host_loads": local_obs["host_loads"],
            "host_ram_usage": local_obs["host_ram_usage"],
            "vm_loads": local_obs["vm_loads"],
            "vm_types": local_obs["vm_types"],
            "vm_available_pes": local_obs["vm_available_pes"],
            "waiting_cloudlets": local_obs["waiting_cloudlets"],
            "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
            "dc_id_onehot": dc_id_onehot,
            "valid_vm_mask": valid_vm_mask,
        }

        # Full action mask
        full_mask = action_mask.astype(np.float32)

        obs_with_mask = {
            "observation": unified_obs,
            "action_mask": full_mask,
        }

        # Get the local policy module
        module = self._rl_module
        if hasattr(module, '__getitem__'):
            module = module[self.policy_id]

        # Prepare batch input
        batch = self._obs_to_batch(obs_with_mask)

        # Forward pass
        with torch.no_grad():
            output = module.forward_inference(batch)

        # Extract action
        if "actions" in output:
            action = output["actions"]
        elif "action_dist_inputs" in output:
            # Apply action mask and argmax
            dist_inputs = output["action_dist_inputs"]
            # Mask invalid actions with large negative value
            mask_tensor = torch.from_numpy(full_mask).unsqueeze(0)
            masked_logits = dist_inputs.clone()
            masked_logits[mask_tensor == 0] = float('-inf')
            action = torch.argmax(masked_logits, dim=-1)
        else:
            raise ValueError(f"Unknown output format: {output.keys()}")

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return int(action.flatten()[0])

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


def create_rllib_new_api_schedulers(algo, env, num_dcs: int, batch_size: int, num_vms: int,
                                     max_hosts: int = 16):
    """
    Create RLlib-based Global and Local schedulers for New API Stack.

    Args:
        algo: loaded RLlib Algorithm instance (New API Stack)
        env: HierarchicalMultiDCEnv instance
        num_dcs: number of datacenters
        batch_size: Global routing batch size
        num_vms: maximum number of VMs per DC
        max_hosts: maximum number of hosts per DC

    Returns:
        (global_scheduler, local_schedulers_dict)
    """
    from .global_schedulers import RLlibNewAPIGlobalScheduler

    global_scheduler = RLlibNewAPIGlobalScheduler(num_dcs, batch_size, algo)

    local_schedulers = {}
    for dc_id in range(num_dcs):
        local_schedulers[dc_id] = RLlibNewAPILocalScheduler(
            num_vms, algo, dc_id, env=env, policy_id="shared_local_policy",
            use_parameter_sharing=True,
            num_datacenters=num_dcs,
            max_hosts=max_hosts,
            max_vms=num_vms,
        )

    return global_scheduler, local_schedulers


# === Register Local Schedulers ===
LOCAL_SCHEDULERS = {
    'random': RandomLocalScheduler,
    'first_fit': FirstFitLocalScheduler,
    'best_fit': BestFitLocalScheduler,
    'worst_fit': WorstFitLocalScheduler,
    'round_robin': RoundRobinLocalScheduler,
    'min_load': MinLoadLocalScheduler,
    'packing_aware': PackingAwareLocalScheduler,
    'ga': GeneticAlgorithmLocalScheduler,
    'pso': ParticleSwarmLocalScheduler,
    # Note: 'rllib' requires special handling via create_rllib_schedulers()
    # Note: 'rllib_new_api' requires special handling via create_rllib_new_api_schedulers()
}
