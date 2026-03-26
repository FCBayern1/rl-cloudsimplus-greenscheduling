"""
debug_ga_obs.py -- 调试观测
src/baselines/evaluate.py -- 基线评估（full/Simple 都用）
tests/verify_action_mask_logic.py, tests/test_reset_gymnasium_compliance.py, tests/test_green_energy.py -- 测试

hierarchical_multidc_pettingzoo.py 直接使用

Hierarchical Multi-Datacenter Reinforcement Learning Environment

This environment implements a two-level hierarchical MARL system:
- Global Level: Routes arriving cloudlets to datacenters (Global Agent)
- Local Level: Schedules cloudlets to VMs within each datacenter (Local Agents)

Architecture:
    Python (Gymnasium) <--> Py4J <--> Java (CloudSim Plus Multi-DC Simulation)
"""

import logging
import time
import subprocess
import socket
import sys
import os
import signal
import atexit
import shutil
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)


# region agent log
def _write_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]):
    try:
        with open("/home/joshua/rl-cloudsimplus-greenscheduling/.cursor/debug-f7b29b.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "f7b29b",
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# endregion


class HierarchicalMultiDCEnv(gym.Env):
    """
    Hierarchical Multi-Datacenter Load Balancing Environment.

    Two-level decision making:
    1. Global Agent: Routes arriving cloudlets to datacenters
    2. Local Agents: Assign cloudlets to VMs within each datacenter

    Action Space:
        - Global: Discrete(num_datacenters) for each arriving cloudlet
        - Local: Discrete(num_vms_per_dc) for each datacenter

    Observation Space:
        - Global: Aggregated state of all datacenters (green power, queues, utilisation)
        - Local: Per-DC state (VM loads, local queues, next cloudlet)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    # Class-level guard to avoid closing the Java gateway multiple times across instances
    _java_gateway_closed = False

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hierarchical multi-datacenter environment.

        Args:
            config: Configuration dictionary containing:
                - multi_datacenter_enabled: bool
                - datacenters: List[dict] of datacenter configurations
                - py4j_port: int (optional, if not provided, a free port is found and a new Java process is launched)
                - global_routing_batch_size: int (cloudlets to route per step, default 5)
                - max_arriving_cloudlets: int (deprecated, for backward compatibility)
                - ... other CloudSim Plus settings
        """
        super(HierarchicalMultiDCEnv, self).__init__()

        self.config = config

        # When True, only build observation/action spaces without launching Java.
        # Used by training scripts that need space shapes for policy construction.
        self._spaces_only = bool(config.get("spaces_only", False))

        # Java Gateway Process Management
        self.java_process = None
        self.py4j_port = config.get("py4j_port")

        if not self._spaces_only:
            if self.py4j_port is None or self.py4j_port == 0:
                self.py4j_port = self._find_free_port()
                self._launch_java_gateway(self.py4j_port)
            else:
                logger.info(f"Using existing Java gateway on port {self.py4j_port}")
        else:
            logger.info("spaces_only mode: skipping Java gateway launch")

        # Cache DC configs and build a stable index <-> dcId mapping.
        # Internally we use dcIndex (0..N-1) for array indexing and observation spaces.
        self.dc_configs = config.get("datacenters")
        if not self.dc_configs:
            self.dc_configs = [{"datacenter_id": 0}]

        self.num_datacenters = len(self.dc_configs)
        self.dc_ids = [
            int(dc.get("datacenter_id", idx)) for idx, dc in enumerate(self.dc_configs)
        ]
        self.dc_id_to_index = {dc_id: idx for idx, dc_id in enumerate(self.dc_ids)}
        self.dc_index_to_id = {idx: dc_id for idx, dc_id in enumerate(self.dc_ids)}
        if len(self.dc_id_to_index) != len(self.dc_ids):
            logger.warning("Duplicate datacenter_id values detected in config; dcId->index mapping may be ambiguous.")
        
        # Fixed batch size for global routing decisions (key parameter)
        self.global_routing_batch_size = config.get("global_routing_batch_size", 10)
        
        # Backward compatibility: if max_arriving_cloudlets is set, use it as batch size
        if "max_arriving_cloudlets" in config and "global_routing_batch_size" not in config:
            logger.warning(
                "'max_arriving_cloudlets' is deprecated. Use 'global_routing_batch_size' instead."
            )
            self.global_routing_batch_size = config.get("max_arriving_cloudlets", 10)

        # Py4J Gateway connection
        self.gateway = None
        self.java_env = None

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Define observation and action spaces
        self._setup_observation_spaces()
        self._setup_action_spaces()

        logger.info(f"HierarchicalMultiDCEnv initialised with {self.num_datacenters} datacenters")
        logger.info(f"  global_routing_batch_size: {self.global_routing_batch_size}")

    def _find_free_port(self) -> int:
        """Find a free TCP port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            return port

    def _launch_java_gateway(self, port: int):
        """
        Launch a dedicated Java CloudSim Plus Gateway process on the specified port.
        """
        # Locate the gradlew script
        # Assuming we are running from the project root or drl-manager
        # Try to find cloudsimplus-gateway directory
        
        possible_roots = [
            os.getcwd(),
            os.path.join(os.getcwd(), ".."),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        ]
        
        gateway_dir = None
        for root in possible_roots:
            candidate = os.path.join(root, "cloudsimplus-gateway")
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "gradlew")):
                gateway_dir = candidate
                break
        
        if not gateway_dir:
            raise RuntimeError("Could not find cloudsimplus-gateway directory with gradlew script.")

        gradlew_path = os.path.join(gateway_dir, "gradlew")
        
        # Prepare command
        # Use --no-daemon to avoid lingering Gradle daemons for each worker
        # Use -q to reduce noise
        cmd = [
            gradlew_path,
            "--no-daemon",
            "-PappMainClass=exe.edu.cspg.MainMultiDC",
            "run",
            "-q",
            f"--args=--port {port}",
        ]
        
        logger.info(f"Launching Java Gateway on port {port}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        # region agent log
        _write_debug_log(
            "H2",
            "hierarchical_multidc_env.py:177",
            "launch_java_gateway",
            {
                "port": port,
                "cwd": os.getcwd(),
                "gateway_dir": gateway_dir,
                "cmd": cmd,
            },
        )
        # endregion
        
        # Launch process
        # Redirect stdout/stderr to a log file for debugging
        log_dir = self.config["gateway_log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"gateway_{port}.log")

        # Serialize gradlew run across Ray workers: concurrent builds in the same
        # cloudsimplus-gateway directory can corrupt build output and cause
        # NoClassDefFoundError (e.g. HierarchicalResetResult) at runtime.
        lock_file = None
        try:
            if sys.platform != "win32":
                lock_path = os.path.join(gateway_dir, ".py4j_gateway_launch.lock")
                lock_file = open(lock_path, "a+", encoding="utf-8")
                logger.info(
                    "Waiting for exclusive gateway build lock (multi-worker safe)..."
                )
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            self._java_log_file = open(log_file_path, "w")

            try:
                self.java_process = subprocess.Popen(
                    cmd,
                    cwd=gateway_dir,
                    stdout=self._java_log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group for easier cleanup
                )

                # Wait for the server to be ready
                # We'll poll the port until it's open
                max_retries = 600  # Gradle cold build + JVM can exceed 60s
                for i in range(max_retries):
                    if self.java_process.poll() is not None:
                        # region agent log
                        _write_debug_log(
                            "H1",
                            "hierarchical_multidc_env.py:201",
                            "java_process_exited_early",
                            {
                                "port": port,
                                "returncode": self.java_process.returncode,
                                "log_file_path": log_file_path,
                            },
                        )
                        # endregion
                        raise RuntimeError(
                            f"Java process exited prematurely with code {self.java_process.returncode}. "
                            f"Check logs at {log_file_path}"
                        )

                    if self._is_port_open(port):
                        logger.info(f"Java Gateway is ready on port {port}")
                        return

                    time.sleep(1.0)
                    if i % 30 == 0 and i > 0:
                        logger.info(
                            f"Waiting for Java Gateway on port {port} ({i}/{max_retries}s)..."
                        )

                raise RuntimeError(f"Timed out waiting for Java Gateway on port {port}")

            except Exception as e:
                logger.error(f"Failed to launch Java Gateway: {e}")
                if self.java_process:
                    self.java_process.kill()
                if self._java_log_file:
                    self._java_log_file.close()
                raise
        finally:
            if lock_file is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_file.close()

    def _is_port_open(self, port: int) -> bool:
        """Check if a TCP port is open and listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, socket.timeout):
                return False

    def _setup_observation_spaces(self):
        """
        Define observation spaces for global and local agents.
        """
        # Global observation space (aggregated DC-level metrics)
        self.global_observation_space = spaces.Dict({
            # Green energy metrics (W - Watts)
            "dc_current_green_power_w": spaces.Box(
                low=0.0, high=5000000.0,  # 5 MW max (increased to accommodate high wind power)
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_current_power_w": spaces.Box(
                low=0.0, high=5_000_000.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_green_ratio": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_cumulative_wasted_green_wh": spaces.Box(
                low=0.0, high=1e6,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            # Future energy trend features (God's Eye mode)
            "dc_future_short_mean": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_short_trend": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_long_mean": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_long_peak_timing": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_queue_sizes": spaces.Box(
                low=0, high=10000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (queue sizes are integers)
            ),
            "dc_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_available_pes": spaces.Box(
                low=0, high=1000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (PEs are integers)
            ),
            "dc_ram_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "upcoming_cloudlets_count": spaces.Discrete(100000),  # Total cloudlets in global waiting queue (increased for large workloads)
            "batch_cloudlet_pes": spaces.Box(
                low=0, high=100,  # Max PEs for a cloudlet
                shape=(self.global_routing_batch_size,),
                dtype=np.int32
            ),
            "batch_cloudlet_mi": spaces.Box(
                low=0, high=1000000,  # Max MI for a cloudlet
                shape=(self.global_routing_batch_size,),
                dtype=np.int64
            ),
            "upcoming_pes_distribution": spaces.Box(
                low=0, high=1000,
                shape=(3,),  # [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]
                dtype=np.int32
            ),
            "load_imbalance": spaces.Box(
                low=0.0, high=10.0,
                shape=(1,),
                dtype=np.float32
            ),
            "recent_completed": spaces.Discrete(100000),  # Increased for large workloads
        })

        # Local observation spaces (per datacenter)
        # Track per-DC sizes but expose a shared max-sized space for SB3 compatibility
        dc_defaults = {
            "hosts_count": 16,
            "initial_s_vm_count": 10,
            "initial_m_vm_count": 5,
            "initial_l_vm_count": 3,
        }
        dc_configs = self.dc_configs or [dc_defaults.copy()]

        self.dc_host_counts: List[int] = [
            int(dc.get("hosts_count", dc_defaults["hosts_count"])) for dc in dc_configs
        ]
        self.max_hosts = max(self.dc_host_counts) if self.dc_host_counts else dc_defaults["hosts_count"]

        self.dc_vm_counts: List[int] = [
            int(
                dc.get("initial_s_vm_count", dc_defaults["initial_s_vm_count"]) +
                dc.get("initial_m_vm_count", dc_defaults["initial_m_vm_count"]) +
                dc.get("initial_l_vm_count", dc_defaults["initial_l_vm_count"])
            )
            for dc in dc_configs
        ]
        self.max_vms = max(self.dc_vm_counts) if self.dc_vm_counts else (
            dc_defaults["initial_s_vm_count"] +
            dc_defaults["initial_m_vm_count"] +
            dc_defaults["initial_l_vm_count"]
        )

        self.local_observation_space = spaces.Dict({
            "host_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "host_ram_usage": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "vm_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_vms,),
                dtype=np.float32
            ),
            "vm_types": spaces.Box(
                low=0, high=3,  # 0=Off, 1=Small, 2=Medium, 3=Large
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "vm_available_pes": spaces.Box(
                low=0, high=100,
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "waiting_cloudlets": spaces.Discrete(100000),  # Increased for large workloads
            "next_cloudlet_pes": spaces.Discrete(256),  # Increased for cloudlets with more PEs
        })

    def _setup_action_spaces(self):
        """
        Define action spaces for global and local agents.
        
        Global Agent: Routes a fixed batch of cloudlets per step.
        - Each action is a datacenter index in [0, num_datacenters - 1]
        - If fewer cloudlets are available than the routing batch size,
          extra actions are simply ignored (trimmed to queue length).
        
        Local Agents: Assign one cloudlet per DC per step.
        - Local action keys are dc_index (0..N-1)
        """
        # Global action space: fixed-size batch of routing decisions.
        # Each element is a DC index ∈ {0, ..., num_datacenters-1}.
        # We no longer use an explicit "NoAssign" action for the global agent;
        # extra actions beyond the current queue length are ignored downstream.
        self.global_action_space = spaces.MultiDiscrete(
            [self.num_datacenters] * self.global_routing_batch_size
        )

        # Local action spaces: Select VM for each datacenter's next cloudlet
        # Each datacenter has its own action space
        # Action space includes: 0 = NoAssign, 1 to max_vms = VM indices
        max_vms = getattr(self, "max_vms", 1)
        self.local_action_space = spaces.Discrete(max_vms + 1)  # +1 for NoAssign option

        # Gymnasium requires self.action_space and self.observation_space
        # Combine global and local spaces into a Dict space
        self.action_space = spaces.Dict({
            "global": self.global_action_space,
            "local": spaces.Dict({
                i: self.local_action_space for i in range(self.num_datacenters)
            })
        })

        self.observation_space = spaces.Dict({
            "global": self.global_observation_space,
            "local": spaces.Dict({
                i: self.local_observation_space for i in range(self.num_datacenters)
            })
        })

    def _connect_to_java(self):
        """
        Establish Py4J connection to Java gateway with retry mechanism.

        Retries connection up to max_retries times with exponential backoff.
        If connection fails after all retries, raises RuntimeError.
        """
        if self.gateway is None:
            max_retries = self.config.get("gateway_max_retries", 5)
            retry_delay = self.config.get("gateway_retry_delay", 5.0)

            logger.info(f"Attempting to connect to Java gateway on port {self.py4j_port}...")

            retries = max_retries
            while retries > 0:
                try:
                    # Attempt connection
                    self.gateway = JavaGateway(
                        gateway_parameters=GatewayParameters(port=self.py4j_port, auto_convert=True)
                    )

                    # Test connection by calling a simple Java method
                    self.gateway.jvm.System.out.println(
                        f"Python HierarchicalMultiDCEnv connected on port {self.py4j_port}!"
                    )

                    self.java_env = self.gateway.entry_point
                    logger.info(f"Successfully connected to Java gateway on port {self.py4j_port}")

                    # Successfully connected, exit retry loop
                    break

                except (ConnectionRefusedError, Py4JNetworkError) as e:
                    retries -= 1
                    if retries > 0:
                        logger.warning(
                            f"Gateway connection failed: {e}. "
                            f"Retrying in {retry_delay} seconds... ({retries} retries left)"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries reached. Could not connect to Java gateway.")
                        raise RuntimeError(
                            f"Could not connect to Java gateway on port {self.py4j_port} "
                            f"after {max_retries} attempts. "
                            f"Make sure the Java gateway server is running:\n"
                            f"  cd cloudsimplus-gateway && ./gradlew run"
                        ) from e

                except Exception as e:
                    # Unexpected error, don't retry
                    logger.error(f"Unexpected error connecting to Java gateway: {e}")
                    raise RuntimeError(
                        f"Unexpected error connecting to Java gateway: {e}"
                    ) from e

            # Configure simulation after successful connection
            try:
                logger.info("Configuring multi-datacenter simulation...")
                self.java_env.configureSimulation(self.config)
                logger.info("Multi-datacenter simulation configured successfully")

            except Exception as e:
                logger.error(f"Failed to configure simulation: {e}")
                # Clean up gateway connection on configuration failure
                self._cleanup_gateway()
                raise RuntimeError(
                    f"Failed to configure multi-datacenter simulation. "
                    f"Check Java logs for details."
                ) from e

    def _cleanup_gateway(self):
        """
        Clean up gateway connection resources.
        """
        if self.gateway is not None:
            try:
                self.gateway.close()
                logger.info("Java gateway connection closed")
            except Exception as e:
                logger.warning(f"Error closing gateway: {e}")
            finally:
                self.gateway = None
                self.java_env = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (not used currently)

        Returns:
            observations: Dict with 'global' and 'local' observations
            info: Additional information

        Raises:
            RuntimeError: If connection to Java gateway fails or reset fails
        """
        super().reset(seed=seed)

        # Connect to Java if not already connected (with retry mechanism)
        self._connect_to_java()

        # Reset Java simulation
        try:
            logger.debug(f"Resetting Java simulation with seed {seed}...")
            result = self.java_env.reset(seed if seed is not None else 0)
        except Exception as e:
            logger.error(f"Failed to reset Java simulation: {e}")
            raise RuntimeError(
                f"Failed to reset multi-datacenter simulation. "
                f"Check Java logs for details."
            ) from e

        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Parse observations from HierarchicalResetResult
        try:
            # Reset returns HierarchicalResetResult (only observations and info)
            observations = self._parse_hierarchical_observation_from_reset(result)
            info = self._parse_info_from_reset(result)
        except Exception as e:
            logger.error(f"Failed to parse reset result: {e}")
            raise RuntimeError(
                f"Failed to parse observations from Java. "
                f"Check observation structure compatibility."
            ) from e

        # Store observations for action masking
        self.last_observations = observations

        logger.info(f"Environment reset successfully for episode (seed={seed})")
        return observations, info

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one hierarchical step in the environment.

        Args:
            action: Dictionary containing:
                - 'global': List of datacenter indices for arriving cloudlets
                - 'local': Dict mapping dc_index -> vm_id

        Returns:
            observations: Dict with 'global' and 'local' observations
            rewards: Dict with 'global' and 'local' rewards
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information

        Raises:
            RuntimeError: If environment not initialized or step execution fails
            ValueError: If action format is invalid
        """
        if self.java_env is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first before calling step()."
            )

        # Validate and extract actions
        try:
            global_actions = action.get("global", [])
            local_actions_map = action.get("local", {})

            if not isinstance(global_actions, (list, np.ndarray)):
                raise ValueError(
                    f"'global' actions must be a list or array, got {type(global_actions)}"
                )
            if not isinstance(local_actions_map, dict):
                raise ValueError(
                    f"'local' actions must be a dict, got {type(local_actions_map)}"
                )

            # Explicit dcIndex usage for local actions (0..N-1).
            # If you have dcId keys, convert them to indices before calling step().
            for raw_key in local_actions_map.keys():
                try:
                    key_int = int(raw_key)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Local action key '{raw_key}' is not an integer dcIndex."
                    )
                if key_int < 0 or key_int >= self.num_datacenters:
                    raise ValueError(
                        f"Local action key '{raw_key}' is out of dcIndex range [0, {self.num_datacenters - 1}]."
                    )
        except Exception as e:
            logger.error(f"Invalid action format: {e}")
            raise ValueError(f"Invalid action format. Expected dict with 'global' and 'local' keys.") from e

        # Process global actions:
        # - Each element is a datacenter index in [0, num_datacenters - 1]
        # - Actions are one-to-one mapped to DC indices; there is no explicit NoAssign.
        # - If there are more actions than available cloudlets, extra actions are ignored.
        # Convert actions to DC indices and clamp out-of-range values
        global_actions_filtered = []
        for i, action_val in enumerate(global_actions):
            action_int = int(action_val)
            if action_int < 0:
                logger.warning(f"Global action[{i}] = {action_int} < 0, clamping to 0")
                dc_index = 0
            elif action_int >= self.num_datacenters:
                logger.warning(
                    f"Global action[{i}] = {action_int} >= {self.num_datacenters}, clamping to {self.num_datacenters - 1}"
                )
                dc_index = self.num_datacenters - 1
            else:
                dc_index = action_int
            global_actions_filtered.append(dc_index)
        
        global_actions = global_actions_filtered

        # Convert local actions dict to Java-compatible format
        # Apply action mapping: agent outputs 0 to num_vms -> Java expects -1 to num_vms-1
        # - action=0 → targetVmId=-1 (NoAssign)
        # - action=1 → targetVmId=0 (VM 0)
        # - action=n → targetVmId=n-1 (VM n-1)
        try:
            # Ensure every DC has an explicit local action; default to NoAssign (0)
            local_actions_java = {}
            for dc_index in range(self.num_datacenters):
                dc_id = self.dc_index_to_id.get(dc_index, dc_index)
                agent_action = local_actions_map.get(dc_index, 0)
                # Map agent action to Java targetVmId
                target_vm_id = int(agent_action) - 1  # 0→-1, 1→0, 2→1, ...
                local_actions_java[int(dc_id)] = target_vm_id
                logger.debug(
                    "DC index %d (dcId=%s): agent_action=%s → targetVmId=%d",
                    dc_index, dc_id, agent_action, target_vm_id
                )
        except Exception as e:
            logger.error(f"Failed to convert local actions: {e}")
            raise ValueError(f"Invalid local action format. DC IDs and VM IDs must be integers.") from e

        # Convert numpy types to Python native types for Py4J compatibility
        # Py4J cannot serialize numpy.int64, numpy.ndarray, etc.
        if isinstance(global_actions, np.ndarray):
            global_actions = global_actions.tolist()
        global_actions_python = [int(x) for x in global_actions]  # Ensure all elements are Python int
        local_actions_python = {int(k): int(v) for k, v in local_actions_java.items()}

        # Execute step in Java simulation
        try:
            logger.info(f"[STEP {self.current_step + 1}] Calling Java with global_actions={global_actions_python}, local_actions={local_actions_python}")
            logger.debug("Calling Java step with %d global actions", len(global_actions_python))
            result = self.java_env.step(global_actions_python, local_actions_python)
            logger.debug("Java step returned successfully")
        except Exception as e:
            logger.error(f"Failed to execute step in Java simulation: {e}")
            logger.debug("Java step FAILED: %s", e)
            raise RuntimeError(
                f"Failed to execute simulation step. Check Java logs for details."
            ) from e

        # Parse results
        try:
            observations = self._parse_hierarchical_observation(result)
            rewards = self._parse_hierarchical_rewards(result)
            terminated = result.isTerminated()
            truncated = result.isTruncated()
            info = self._parse_info(result)
        except Exception as e:
            logger.error(f"Failed to parse step result: {e}")
            raise RuntimeError(
                f"Failed to parse step results from Java. "
                f"Check observation/reward structure compatibility."
            ) from e

        # Update episode state
        self.current_step += 1
        self.episode_reward += rewards["global"]
        self.done = terminated or truncated

        # Store observations for action masking
        self.last_observations = observations

        logger.debug(
            f"Step {self.current_step}: Global reward={rewards['global']:.3f}, "
            f"Terminated={terminated}, Truncated={truncated}"
        )

        return observations, rewards, terminated, truncated, info

    def _parse_hierarchical_observation_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalResetResult into observation dict.
        This is specifically for reset() which returns HierarchicalResetResult.
        """
        # Parse global observation (GlobalObservationState)
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations (Map<Integer, ObservationState>)
        local_obs_java = result.getLocalObservations()
        local_obs = {}
        for dc_id in local_obs_java:
            try:
                obs_state = (
                    local_obs_java.get(dc_id)
                    if hasattr(local_obs_java, "get")
                    else local_obs_java[dc_id]
                )
            except Exception:
                obs_state = None
            if obs_state is not None:
                dc_id_int = int(dc_id)
                dc_index = self.dc_id_to_index.get(dc_id_int)
                if dc_index is None:
                    logger.warning("Unknown datacenter_id in reset observations: %s", dc_id_int)
                    continue
                local_obs[dc_index] = self._convert_local_observation(dc_index, obs_state)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_info_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse info from HierarchicalResetResult.
        This is specifically for reset() which returns HierarchicalResetResult.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()
        info = {}
        # Be robust to Py4J Map proxies vs auto-converted Python dicts.
        # Prefer keySet()/get() when available.
        try:
            if hasattr(info_java, "keySet") and hasattr(info_java, "get"):
                for key in info_java.keySet():
                    value = info_java.get(key)
                    info[str(key)] = self._convert_java_value(value)
                return info
        except Exception:
            pass

        # Fallback: assume it behaves like a Python mapping / iterable of keys
        for key in info_java:
            try:
                value = info_java[key]
            except Exception:
                # Last resort: try .get(key) without default
                value = info_java.get(key) if hasattr(info_java, "get") else None
            info[str(key)] = self._convert_java_value(value)
        return info

    def _pad_batch_array(self, arr: np.ndarray, target_size: int, dtype=np.int32) -> np.ndarray:
        """
        Pad or trim array to match target_size.

        Args:
            arr: Input array from Java gateway
            target_size: Target array size (global_routing_batch_size)
            dtype: Array data type

        Returns:
            Array of exactly target_size elements
        """
        if len(arr) >= target_size:
            # Trim to target size
            return arr[:target_size]
        else:
            # Pad with zeros
            result = np.zeros(target_size, dtype=dtype)
            result[:len(arr)] = arr
            return result

    def _convert_global_observation(self, global_obs_java) -> Dict[str, Any]:
        """
        Convert Java GlobalObservationState to Python dict.
        """
        return {
            # Green energy metrics
            "dc_current_green_power_w": np.array(global_obs_java.getDcCurrentGreenPowerW(), dtype=np.float32),
            "dc_current_power_w": np.array(global_obs_java.getDcCurrentPowerW(), dtype=np.float32),
            "dc_green_ratio": np.array(global_obs_java.getDcGreenRatio(), dtype=np.float32),
            "dc_cumulative_wasted_green_wh": np.array(global_obs_java.getDcCumulativeWastedGreenWh(), dtype=np.float32),
            # Future energy trend features (God's Eye mode)
            "dc_future_short_mean": np.array(global_obs_java.getDcFutureShortMean(), dtype=np.float32),
            "dc_future_short_trend": np.array(global_obs_java.getDcFutureShortTrend(), dtype=np.float32),
            "dc_future_long_mean": np.array(global_obs_java.getDcFutureLongMean(), dtype=np.float32),
            "dc_future_long_peak_timing": np.array(global_obs_java.getDcFutureLongPeakTiming(), dtype=np.float32),
            # Resource metrics
            "dc_queue_sizes": np.array(global_obs_java.getDcQueueSizes(), dtype=np.int32),
            "dc_utilizations": np.array(global_obs_java.getDcUtilizations(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getDcAvailablePes(), dtype=np.int32),
            "dc_ram_utilizations": np.array(global_obs_java.getDcRamUtilizations(), dtype=np.float32),
            # Clamp Discrete values to valid range to prevent one_hot errors
            "upcoming_cloudlets_count": min(global_obs_java.getUpcomingCloudletsCount(), 99999),
            "batch_cloudlet_pes": self._pad_batch_array(
                np.array(global_obs_java.getBatchCloudletPes(), dtype=np.int32),
                self.global_routing_batch_size, dtype=np.int32
            ),
            "batch_cloudlet_mi": self._pad_batch_array(
                np.array(global_obs_java.getBatchCloudletMi(), dtype=np.int64),
                self.global_routing_batch_size, dtype=np.int64
            ),
            "upcoming_pes_distribution": np.array(global_obs_java.getUpcomingCloudletsPesDistribution(), dtype=np.int32),
            "load_imbalance": np.array([global_obs_java.getLoadImbalance()], dtype=np.float32),
            "recent_completed": min(global_obs_java.getRecentCompletedCloudlets(), 99999),
        }

    def _convert_local_observation(self, dc_id: int, local_obs_java) -> Dict[str, Any]:
        """
        Convert Java ObservationState to Python dict, padding/trimming so each DC
        matches the shared observation space while preserving its own host/VM count.
        """
        host_target = self._get_dc_host_count(dc_id)
        vm_target = self._get_dc_vm_count(dc_id)

        host_loads = np.array(local_obs_java.getHostLoads(), dtype=np.float32)[:host_target]
        host_ram_usage = np.array(local_obs_java.getHostRamUsageRatio(), dtype=np.float32)[:host_target]
        vm_loads = np.array(local_obs_java.getVmLoads(), dtype=np.float32)[:vm_target]
        vm_types = np.array(local_obs_java.getVmTypes(), dtype=np.int32)[:vm_target]
        vm_available_pes = np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32)[:vm_target]

        return {
            "host_loads": self._pad_vector(host_loads, self.max_hosts, 0.0),
            "host_ram_usage": self._pad_vector(host_ram_usage, self.max_hosts, 0.0),
            "vm_loads": self._pad_vector(vm_loads, self.max_vms, 0.0),
            "vm_types": self._pad_vector(vm_types, self.max_vms, 0),
            "vm_available_pes": self._pad_vector(vm_available_pes, self.max_vms, 0),
            # Clamp Discrete values to valid range to prevent one_hot errors
            "waiting_cloudlets": min(local_obs_java.getWaitingCloudlets(), 99999),
            "next_cloudlet_pes": min(local_obs_java.getNextCloudletPes(), 255),
        }

    def _get_dc_host_count(self, dc_id: int) -> int:
        """Return configured host count for a datacenter (fallback to max_hosts)."""
        if hasattr(self, "dc_host_counts") and 0 <= dc_id < len(self.dc_host_counts):
            return self.dc_host_counts[dc_id]
        return getattr(self, "max_hosts", 1)

    def _get_dc_vm_count(self, dc_id: int) -> int:
        """Return configured VM count for a datacenter (fallback to max_vms)."""
        if hasattr(self, "dc_vm_counts") and 0 <= dc_id < len(self.dc_vm_counts):
            return self.dc_vm_counts[dc_id]
        return getattr(self, "max_vms", 1)

    @staticmethod
    def _pad_vector(values: np.ndarray, target_len: int, fill_value: float) -> np.ndarray:
        """
        Ensure vectors share a consistent length by trimming overflow and padding
        the tail with a provided fill_value.
        """
        current_len = values.shape[0]
        if current_len == target_len:
            return values

        if current_len > target_len:
            return values[:target_len]

        padded = np.full((target_len,), fill_value, dtype=values.dtype)
        if current_len > 0:
            padded[:current_len] = values
        return padded

    def _parse_hierarchical_observation(
        self,
        result  # HierarchicalStepResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalStepResult into observation dict.
        This is specifically for step() which returns HierarchicalStepResult.
        """
        # Parse global observation
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations
        local_obs_map_java = result.getLocalObservations()
        local_obs = {}
        if local_obs_map_java is None:
            return {"global": global_obs, "local": local_obs}

        # Be robust to Py4J Map proxies vs auto-converted Python dicts:
        # iterate actual keys provided by Java rather than assuming 0..N-1 membership works.
        try:
            keys_iter = local_obs_map_java.keySet() if hasattr(local_obs_map_java, "keySet") else local_obs_map_java
            for dc_id_raw in keys_iter:
                dc_id = int(dc_id_raw)
                dc_index = self.dc_id_to_index.get(dc_id)
                if dc_index is None:
                    logger.warning("Unknown datacenter_id in step observations: %s", dc_id)
                    continue
                try:
                    obs_state = (
                        local_obs_map_java.get(dc_id_raw)
                        if hasattr(local_obs_map_java, "get")
                        else local_obs_map_java[dc_id_raw]
                    )
                except Exception:
                    # Fallback: attempt Python-int lookup
                    obs_state = (
                        local_obs_map_java.get(dc_id)
                        if hasattr(local_obs_map_java, "get")
                        else local_obs_map_java[dc_id]
                    )

                if obs_state is not None:
                    local_obs[dc_index] = self._convert_local_observation(dc_index, obs_state)
        except Exception as e:
            logger.error("Failed to parse local observations map: %s", e)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_hierarchical_rewards(
        self,
        result
    ) -> Dict[str, Any]:
        """
        Parse hierarchical rewards from step result.

        Returns:
            {
                'global': float,
                'local': {dc_index: float}
            }
        """
        global_reward = result.getGlobalReward()

        local_rewards_java = result.getLocalRewards()
        local_rewards = {}
        for dc_index in range(self.num_datacenters):
            dc_id = self.dc_index_to_id.get(dc_index, dc_index)
            try:
                if hasattr(local_rewards_java, "get"):
                    reward_val = local_rewards_java.get(dc_id, 0.0)
                else:
                    reward_val = local_rewards_java[dc_id]
            except Exception:
                reward_val = 0.0
            local_rewards[dc_index] = reward_val

        return {
            "global": global_reward,
            "local": local_rewards
        }

    def _parse_info(self, result) -> Dict[str, Any]:
        """
        Parse additional info from step result.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()

        # Convert Java Map to Python dict with serializable values
        info = {}
        # Be robust to Py4J Map proxies vs auto-converted Python dicts.
        # Prefer keySet()/get() when available.
        try:
            if hasattr(info_java, "keySet") and hasattr(info_java, "get"):
                for key in info_java.keySet():
                    value = info_java.get(key)
                    info[str(key)] = self._convert_java_value(value)
                info["episode_step"] = self.current_step
                info["episode_reward"] = self.episode_reward
                return info
        except Exception:
            pass

        # Fallback: assume it behaves like a Python mapping / iterable of keys
        for key in info_java:
            try:
                value = info_java[key]
            except Exception:
                value = info_java.get(key) if hasattr(info_java, "get") else None
            info[str(key)] = self._convert_java_value(value)

        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward

        return info
    
    def _convert_java_value(self, value):
        """
        Convert a Java value (from Py4J) to a Python native type.

        IMPORTANT:
        - Preserves nested Maps/Lists by converting them recursively to dict/list
        - Avoids stringifying complex objects such as energy metrics maps
        """
        if value is None:
            return None

        # Already plain Python scalar
        if isinstance(value, (bool, int, float, str)):
            return value

        # Handle Java Maps (e.g., HashMap) exposed via Py4J:
        # They usually have keySet() and get() methods.
        try:
            if hasattr(value, "keySet") and hasattr(value, "get"):
                py_dict = {}
                for k in value.keySet():
                    # Try to keep numeric keys as int (e.g., DC id 0..N-1),
                    # fall back to string for non-numeric keys.
                    try:
                        py_key = int(k)
                    except (TypeError, ValueError):
                        py_key = str(k)
                    py_dict[py_key] = self._convert_java_value(value.get(k))
                return py_dict
        except Exception:
            # If anything goes wrong, fall through to other heuristics
            pass

        # Handle Java Lists or other iterable collections
        try:
            # Many Py4J Java collections are iterable but not sequences
            iterator = iter(value)
        except TypeError:
            iterator = None

        if iterator is not None:
            try:
                return [self._convert_java_value(v) for v in list(iterator)]
            except Exception:
                # If iteration fails, continue to scalar conversion attempts
                pass

        # Try numeric conversions (Integer, Long, Double, etc.)
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        # Try boolean from string representation
        try:
            s = str(value).lower()
            if s in ("true", "false"):
                return s == "true"
        except Exception:
            pass

        # Fallback: string representation for unknown complex types
        return str(value)

    def render(self):
        """
        Render the environment (not implemented for this environment).
        """
        pass

    def close(self):
        """
        Close the environment and cleanup resources.

        Safely closes the Java simulation environment and shuts down the Py4J gateway.
        This method is called automatically by Gymnasium when the environment is no longer needed.
        """
        # Close Java simulation environment
        if self.java_env is not None:
            try:
                logger.info("Closing Java simulation environment...")
                self.java_env.close()
                logger.info("Java simulation environment closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Java simulation environment: {e}")

        # Close Py4J gateway client connection (do NOT shutdown the Java server,
        # so that multiple evaluations / combinations can reuse the same JVM)
        if self.gateway is not None:
            try:
                logger.info("Closing Py4J gateway client connection...")
                self.gateway.close()
                logger.info("Py4J gateway client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Py4J gateway client: {e}")
            finally:
                self.gateway = None
                self.java_env = None
        
        # Terminate the Java process if we launched it
        if self.java_process:
            try:
                logger.info(f"Terminating Java Gateway process (PID {self.java_process.pid})...")
                os.killpg(os.getpgid(self.java_process.pid), signal.SIGTERM)
                self.java_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating Java process: {e}")
                try:
                    os.killpg(os.getpgid(self.java_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            finally:
                self.java_process = None
                if hasattr(self, '_java_log_file') and self._java_log_file:
                    self._java_log_file.close()

    def get_num_datacenters(self) -> int:
        """Get the number of datacenters in the environment."""
        return self.num_datacenters

    def get_arriving_cloudlets_count(self) -> int:
        """
        Get the number of cloudlets in global waiting queue.
        
        DEPRECATED: This method name is misleading. Use get_global_waiting_cloudlets_count() instead.
        Kept for backward compatibility with tests.
        """
        return self.get_global_waiting_cloudlets_count()
    
    def get_global_waiting_cloudlets_count(self) -> int:
        """Get the number of cloudlets in the global waiting queue (batch routing mode)."""
        if self.java_env is None:
            return 0
        return self.java_env.getGlobalWaitingCloudletsCount()

    def get_global_action_mask(self, global_obs: Dict[str, Any]) -> np.ndarray:
        """
        Generate slot-level mask for global MultiDiscrete routing action.

        Mask semantics:
        - mask[i] = 1.0: slot i corresponds to a real cloudlet in the upcoming batch
        - mask[i] = 0.0: slot i is padding (no cloudlet)

        Preferred source is batch_cloudlet_pes/mi arrays (direct slot-level signals).
        Fallback uses upcoming_cloudlets_count to derive a valid prefix length.

        Args:
            global_obs: Global observation dict.

        Returns:
            np.ndarray of shape (global_routing_batch_size,), dtype float32.
        """
        batch_size = int(self.global_routing_batch_size)
        if batch_size <= 0:
            return np.zeros((0,), dtype=np.float32)

        try:
            pes = np.asarray(global_obs.get("batch_cloudlet_pes", []), dtype=np.int64)
            mi = np.asarray(global_obs.get("batch_cloudlet_mi", []), dtype=np.int64)

            # Use slot-level batch features when available.
            if pes.size > 0 and mi.size > 0:
                use_len = min(batch_size, int(pes.size), int(mi.size))
                mask = np.zeros(batch_size, dtype=np.float32)
                if use_len > 0:
                    valid_slots = (pes[:use_len] > 0) & (mi[:use_len] > 0)
                    mask[:use_len] = valid_slots.astype(np.float32)
                return mask
        except Exception as e:
            logger.debug("Failed to build global action mask from batch arrays: %s", e)

        # Fallback: derive valid prefix from queue count.
        try:
            upcoming_count = int(global_obs.get("upcoming_cloudlets_count", 0))
            valid_len = max(0, min(batch_size, upcoming_count))
            mask = np.zeros(batch_size, dtype=np.float32)
            mask[:valid_len] = 1.0
            return mask
        except Exception as e:
            logger.warning(
                "Failed to build global action mask from upcoming_cloudlets_count (%s). "
                "Allowing all global action slots.", e
            )
            return np.ones(batch_size, dtype=np.float32)

    def get_local_action_masks(self, dc_id: int) -> np.ndarray:
        """
        Generate action mask for a specific datacenter's local agent.

        Mask logic (consistent with Single-DC environment):
        - If queue is empty: only allow action 0 (NoAssign)
        - If queue has tasks: forbid action 0, allow VMs with enough resources
        - If no VM has enough resources: allow all VMs (Java handles penalty)

        Args:
            dc_id: Datacenter index (0..N-1). Must be an index.

        Returns:
            mask: Boolean array of shape (num_vms+1,) where True = action allowed
        """
        # Enforce explicit dcIndex usage to avoid id/index ambiguity.
        dc_index = dc_id
        if dc_index < 0 or dc_index >= self.num_datacenters:
            if dc_id in self.dc_id_to_index:
                raise ValueError(
                    f"get_local_action_masks expects dcIndex. Received dcId={dc_id}. "
                    "Convert to dcIndex before calling."
                )
            raise ValueError(
                f"get_local_action_masks expects dcIndex in [0, {self.num_datacenters - 1}]. "
                f"Received {dc_id}."
            )

        # Fallback: allow all actions if environment not initialized or invalid index
        if self.java_env is None or dc_index >= self.num_datacenters or dc_index < 0:
            logger.warning(f"Cannot generate mask for DC {dc_id}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get DC state from last observation
        try:
            if not hasattr(self, 'last_observations') or 'local' not in self.last_observations:
                logger.debug(f"No observations available yet, allowing all actions for DC {dc_id}")
                return np.ones(self.local_action_space.n, dtype=bool)

            local_obs = self.last_observations["local"].get(dc_index)
            if local_obs is None:
                logger.warning(f"No observation for DC {dc_id}, allowing all actions")
                return np.ones(self.local_action_space.n, dtype=bool)

            vm_available_pes = local_obs["vm_available_pes"]
            waiting_cloudlets = local_obs["waiting_cloudlets"]
            next_cloudlet_pes = local_obs["next_cloudlet_pes"]

        except Exception as e:
            logger.error(f"Failed to extract state for DC {dc_id}: {e}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get actual VM count for this DC
        dc_vm_count = self._get_dc_vm_count(dc_index)
        
        # Initialize mask (all False)
        mask = np.zeros(self.local_action_space.n, dtype=bool)

        # Case 1: Queue is empty or next task invalid
        if waiting_cloudlets == 0 or next_cloudlet_pes == 0:
            mask[0] = True  # Only allow action 0 (NoAssign)
            logger.debug(f"DC {dc_id}: Queue empty, only NoAssign allowed")
            return mask

        # Case 2: Queue has tasks
        mask[0] = False  # Forbid explicit NoAssign (encourage assignment)

        # Check each VM's resources (only actual VMs, not padding)
        has_valid_vm = False
        for vm_idx in range(min(len(vm_available_pes), dc_vm_count)):
            available_pes = vm_available_pes[vm_idx]
            if available_pes >= next_cloudlet_pes:
                mask[vm_idx + 1] = True  # action (vm_idx+1) → targetVmId (vm_idx)
                has_valid_vm = True

        # Case 3: No VM has enough resources
        # Align with loadbalancing_env.py: Force assignment (disallow NoAssign)
        if not has_valid_vm:
            logger.debug(f"DC {dc_id}: No VM has {next_cloudlet_pes} PEs, allowing all VMs (forcing assignment)")
            mask[0] = False  # Disallow NoAssign
            mask[1:dc_vm_count+1] = True  # Allow all VMs

        logger.debug(f"DC {dc_id}: Mask generated - {np.sum(mask)}/{len(mask)} actions allowed")
        return mask
