# RL-CloudSim Load Balancing System Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Py4J Communication Bridge](#4-py4j-communication-bridge)
5. [Key Components Explained](#5-key-components-explained)
6. [Observation and Action Spaces](#6-observation-and-action-spaces)
7. [Reward Function Design](#7-reward-function-design)
8. [Complete Workflow](#8-complete-workflow)
9. [Training Process](#9-training-process)
10. [Project Structure](#10-project-structure)
11. [Configuration System](#11-configuration-system)
12. [How to Run](#12-how-to-run)
13. [Hierarchical Multi-Datacenter System](#13-hierarchical-multi-datacenter-system)
14. [Extending the System](#14-extending-the-system)

---

## 1. Project Overview

### What is This Project?

This project is a **Reinforcement Learning-based Cloud Task Scheduling System** that uses:
- **CloudSim Plus** (Java): A cloud datacenter simulation framework
- **Stable-Baselines3** (Python): State-of-the-art RL algorithms (PPO, RecurrentPPO)
- **Py4J**: A bridge that enables Python and Java to communicate

### Why Two Languages?

- **Java**: CloudSim Plus is a mature, high-performance discrete-event simulator written in Java. It accurately models datacenters, VMs, hosts, and task scheduling.
- **Python**: The deep learning and RL ecosystem (PyTorch, TensorFlow, Stable-Baselines3) is predominantly Python-based.
- **Solution**: Use Py4J to connect them seamlessly.

### High-Level Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop (Python)                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Agent observes state (VM loads, queue info)     │    │
│  │ 2. Agent selects action (which VM to assign task)  │    │
│  │ 3. Environment executes action via Py4J            │────┼──┐
│  │ 4. Simulation advances, returns reward + new state │◄───┼──┘
│  │ 5. Agent learns from experience (PPO update)       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Py4J Bridge (TCP/IP)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CloudSim Plus Simulation (Java)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ • Manages datacenter infrastructure (Hosts, VMs)   │    │
│  │ • Processes cloudlet (task) arrivals              │    │
│  │ • Executes scheduling decisions                    │    │
│  │ • Calculates resource utilization & wait times    │    │
│  │ • Tracks energy consumption                        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture

### Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         PYTHON LAYER                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐         ┌──────────────────┐               │
│  │  Training Script │────────►│ LoadBalancingEnv │               │
│  │   (train.py)     │         │ (Gym Environment)│               │
│  └─────────────────┘         └──────────────────┘               │
│          │                            │                           │
│          │ Creates & Trains           │ Wraps Java Simulation    │
│          ▼                            ▼                           │
│  ┌─────────────────┐         ┌──────────────────┐               │
│  │ RecurrentPPO    │         │   JavaGateway    │               │
│  │ (SB3 Algorithm) │         │   (Py4J Client)  │               │
│  └─────────────────┘         └──────────────────┘               │
│                                       │                           │
└───────────────────────────────────────┼───────────────────────────┘
                                        │ TCP/IP (Port 25333)
                                        │
┌───────────────────────────────────────┼───────────────────────────┐
│                         JAVA LAYER    │                           │
├───────────────────────────────────────┼───────────────────────────┤
│                                       ▼                           │
│                            ┌──────────────────────┐              │
│                            │   GatewayServer      │              │
│                            │   (Py4J Server)      │              │
│                            └──────────────────────┘              │
│                                       │                           │
│                                       │ Exposes                   │
│                                       ▼                           │
│                       ┌──────────────────────────────┐           │
│                       │  LoadBalancerGateway         │           │
│                       │  (Entry Point for Python)    │           │
│                       └──────────────────────────────┘           │
│                                       │                           │
│                    ┌──────────────────┼──────────────────┐       │
│                    │                  │                  │       │
│                    ▼                  ▼                  ▼       │
│          ┌─────────────────┐  ┌─────────────┐  ┌──────────────┐│
│          │ SimulationCore  │  │ObservationS │  │Reward Calc.  ││
│          │                 │  │tate Builder │  │              ││
│          └─────────────────┘  └─────────────┘  └──────────────┘│
│                    │                                             │
│                    ▼                                             │
│          ┌─────────────────────────────────────┐                │
│          │  CloudSim Plus Framework            │                │
│          │  • DatacenterSimulation             │                │
│          │  • LoadBalancingBroker              │                │
│          │  • Hosts, VMs, Cloudlets            │                │
│          │  • Event Scheduling                 │                │
│          └─────────────────────────────────────┘                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Language | Responsibility |
|-----------|----------|----------------|
| **train.py** | Python | Orchestrates training loop, loads config, creates environment, runs RL algorithm |
| **LoadBalancingEnv** | Python | Gym environment wrapper, translates between SB3 and Java |
| **JavaGateway (Py4J)** | Python | Client that connects to Java, sends method calls over TCP |
| **GatewayServer (Py4J)** | Java | Server listening on port 25333, receives method calls from Python |
| **LoadBalancerGateway** | Java | Main entry point for Python, coordinates simulation steps |
| **SimulationCore** | Java | Manages CloudSim Plus simulation lifecycle |
| **LoadBalancingBroker** | Java | Custom broker that waits for agent decisions before scheduling |
| **ObservationState** | Java | Data container for observations sent to Python |
| **SimulationStepInfo** | Java | Data container for step results (obs, reward, done, info) |

---

## 3. Technology Stack

### Python Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **stable-baselines3** | 2.0+ | RL algorithms (PPO, A2C) |
| **sb3-contrib** | 2.0+ | Additional algorithms (RecurrentPPO, MaskablePPO) |
| **gymnasium** | 0.29+ | Standard RL environment interface |
| **py4j** | 0.10.9+ | Java-Python bridge |
| **numpy** | 1.24+ | Numerical computations |
| **torch** | 2.0+ | Deep learning backend for SB3: **RTX 50 series GPU** need the following version: pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 |
| **tensorboard** | 2.14+ | Training visualization |
| **pyyaml** | 6.0+ | Configuration file parsing |

### Java Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **CloudSim Plus** | 8.0.0+ | Datacenter simulation framework |
| **Py4J** | 0.10.9.7 | Java-Python bridge |
| **SLF4J** | 2.0.9 | Logging framework |
| **Logback** | 1.4.11 | Logging implementation |

### Build Tools

- **Gradle 8.5**: Java build system
- **Python 3.10+**: Python runtime

---

## 4. Py4J Communication Bridge

### What is Py4J?

**Py4J** is a library that enables Python programs to dynamically access Java objects in a Java Virtual Machine (JVM). Unlike JNI (Java Native Interface), Py4J doesn't require native code compilation.

**Key Features:**
- Bidirectional communication (Python ↔ Java)
- Automatic type conversion
- Method calls over TCP/IP
- No code generation required

### How Py4J Works Internally

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Process (Client)                       │
│                                                                   │
│  Python Code:                                                    │
│  >>> gateway = JavaGateway()                                     │
│  >>> result = gateway.entry_point.step(action)                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Py4J Client Library (JavaGateway)                  │         │
│  │                                                     │         │
│  │  1. Serialize method call:                         │         │
│  │     "call entry_point.step with params [action]"   │         │
│  │                                                     │         │
│  │  2. Send over TCP socket to localhost:25333        │         │
│  │                                                     │         │
│  │  3. Wait for response                              │         │
│  │                                                     │         │
│  │  4. Deserialize response:                          │         │
│  │     - Convert Java objects to Python proxies       │         │
│  │     - Convert primitives to Python types           │         │
│  └────────────────────────────────────────────────────┘         │
│                          │ ▲                                     │
└──────────────────────────┼─┼─────────────────────────────────────┘
                           │ │ TCP/IP Connection
                           │ │ (localhost:25333)
┌──────────────────────────▼─┼─────────────────────────────────────┐
│                    Java Process (Server)                         │
│                            │ │                                    │
│  ┌────────────────────────┼─┼─────────────────────────┐         │
│  │ Py4J Server Library (GatewayServer)                │         │
│  │                        │ │                          │         │
│  │  1. Listen on port 25333│ │                         │         │
│  │                         │ │                         │         │
│  │  2. Receive method call │ │                         │         │
│  │                         │ │                         │         │
│  │  3. Parse and execute:  │ │                         │         │
│  │     entryPoint.step(action)                         │         │
│  │                         │ │                         │         │
│  │  4. Serialize result:   │ │                         │         │
│  │     - Convert Java objects to transferable format  │         │
│  │     - Send back over socket                         │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│                ┌─────────────────────────────┐                   │
│                │  LoadBalancerGateway        │                   │
│                │  (Entry Point Object)       │                   │
│                │                             │                   │
│                │  public SimulationStepInfo  │                   │
│                │  step(int action) {         │                   │
│                │      // Execute action      │                   │
│                │      return stepInfo;       │                   │
│                │  }                          │                   │
│                └─────────────────────────────┘                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Step-by-Step: How Observations Flow from Java to Python

Let's trace a complete observation transfer:

#### Step 1: Java Creates Observation Object

```java
// In LoadBalancerGateway.java
ObservationState obs = getCurrentState();

// ObservationState contains:
double[] vmLoads = {0.45, 0.78, 0.12};  // 3 VMs
int[] vmTypes = {1, 2, 3};               // Small, Medium, Large
int waitingCloudlets = 5;
long nextCloudletMi = 7234567L;
// ... etc
```

#### Step 2: Java Returns Object via Py4J

```java
// In LoadBalancerGateway.java
return new SimulationStepInfo(
    obs,           // ← ObservationState object
    reward,
    terminated,
    truncated,
    infoMap
);
```

When this method returns, Py4J intercepts the return value.

#### Step 3: Py4J Serializes and Transfers

Py4J doesn't send the actual Java object over the network. Instead:

**For primitive types and arrays:**
- Automatically converted and sent as values
- `int` → Python `int`
- `double[]` → Python `list[float]`
- `long` → Python `int`

**For custom objects (like ObservationState):**
- Py4J creates a **proxy reference**
- Python receives a `JavaObject` proxy
- The actual object stays in Java memory
- Method calls on the proxy are forwarded back to Java

#### Step 4: Python Receives Proxy Object

```python
# In loadbalancing_env.py
step_result_java = self.loadbalancer_gateway.step(target_vm_id)

# step_result_java is a JavaObject proxy
# It's NOT a Python object, it's a reference to the Java object
```

#### Step 5: Python Extracts Data via Getter Methods

```python
# Each getter call is a network round-trip!
java_obs_state = step_result_java.getObservation()  # Returns another JavaObject proxy

# Now extract actual data
vm_loads_list = java_obs_state.getVmLoads()  # Py4J converts double[] → list
vm_types_list = java_obs_state.getVmTypes()   # Py4J converts int[] → list
waiting = java_obs_state.getWaitingCloudlets()  # Py4J converts int → int
next_mi = java_obs_state.getNextCloudletMi()    # Py4J converts long → int

# Convert to NumPy arrays
vm_loads = np.array(vm_loads_list, dtype=np.float32)
vm_types = np.array(vm_types_list, dtype=np.int32)
```

### Step-by-Step: How Actions Flow from Python to Java

#### Step 1: Agent Selects Action

```python
# In train.py (handled by SB3)
model.learn(total_timesteps):
    for step in range(n_steps):
		action = model.predict(observation)  # Returns integer, e.g., 2
        obs, reward, done, info = env.step(action) 
        rollout_buffer.add(obs, action, reward, done)

# action is interpreted as:
# 0 = No-op (don't assign task)
# 1 = Assign to VM 0
# 2 = Assign to VM 1
# 3 = Assign to VM 2
# etc.
```

#### Step 2: Environment Maps Action to VM ID

```python
# In loadbalancing_env.py step() method
target_vm_id = int(action) - 1

# Examples:
# action=0 → target_vm_id=-1 (No-op)
# action=1 → target_vm_id=0 (VM 0)
# action=2 → target_vm_id=1 (VM 1)
```

#### Step 3: Python Calls Java Method via Py4J

```python
step_result_java = self.loadbalancer_gateway.step(target_vm_id)
```

This triggers:
1. Py4J serializes the method call and parameter (`target_vm_id`)
2. Sends over TCP to Java
3. Java deserializes and invokes `LoadBalancerGateway.step(int vmId)`

#### Step 4: Java Executes Action

```java
// In LoadBalancerGateway.java
public SimulationStepInfo step(int vmId) {
    if (vmId < 0) {
        // No-op action
        isNoOpAction = true;
    } else {
        // Assign next cloudlet to specified VM
        Cloudlet cloudlet = broker.getNextPendingCloudlet();
        Vm targetVm = getVmById(vmId);
        broker.assignCloudletToVm(cloudlet, targetVm);
    }

    // Advance simulation
    simulation.runFor(SCHEDULING_INTERVAL);

    // Calculate reward and next observation
    return new SimulationStepInfo(...);
}
```

#### Step 5: Java Returns Result to Python

The result flows back through Py4J (see observation flow above).

### Type Conversion Reference

| Java Type | Py4J Conversion | Python Type |
|-----------|----------------|-------------|
| `int` | Automatic | `int` |
| `long` | Automatic | `int` |
| `double` | Automatic | `float` |
| `boolean` | Automatic | `bool` |
| `String` | Automatic | `str` |
| `int[]` | Automatic | `list[int]` |
| `double[]` | Automatic | `list[float]` |
| `List<Integer>` | Automatic | `list[int]` |
| `Map<String, Object>` | Automatic | `dict` |
| Custom Object | **Proxy** | `JavaObject` (requires getters) |

### Performance Considerations

**Each method call over Py4J has overhead:**
- TCP round-trip latency (~0.1-1ms on localhost)
- Serialization/deserialization cost

**Best Practices:**
1. **Batch data**: Return arrays instead of calling getters in loops
2. **Use primitives**: Prefer `double[]` over `List<Double>` for auto-conversion
3. **Minimize calls**: Fetch all needed data in one object
4. **Keep JVM warm**: Don't restart Java process frequently

**Example - Really High Overhead (many round-trips):**

```python
# DON'T DO THIS
for i in range(num_vms):
    load = java_obs.getVmLoad(i)  # One network call per VM!
```

**Example - Good (one round-trip):**
```python
# DO THIS
all_loads = java_obs.getVmLoads()  # One network call, returns array
```

### 

### Py4J Communication Sequence Diagram

```
Python (Client)                    Py4J Bridge                    Java (Server)
      │                                  │                                │
      │  gateway = JavaGateway()         │                                │
      ├─────────────────────────────────►│                                │
      │                                  │  Connect to localhost:25333    │
      │                                  ├───────────────────────────────►│
      │                                  │                                │ GatewayServer
      │                                  │◄───────────────────────────────┤ listening
      │  Connection established          │                                │
      │◄─────────────────────────────────┤                                │
      │                                  │                                │
      │  entry = gateway.entry_point     │                                │
      ├─────────────────────────────────►│                                │
      │                                  │  Get entry point object        │
      │                                  ├───────────────────────────────►│
      │                                  │◄───────────────────────────────┤
      │  Return proxy to                 │   Return LoadBalancerGateway   │
      │  LoadBalancerGateway             │                                │
      │◄─────────────────────────────────┤                                │
      │                                  │                                │
      │  result = entry.step(2)          │                                │
      ├─────────────────────────────────►│                                │
      │                                  │  Invoke step(2) on             │
      │                                  │  LoadBalancerGateway           │
      │                                  ├───────────────────────────────►│
      │                                  │                                │ Execute
      │                                  │                                │ step(2)
      │                                  │                                │
      │                                  │◄───────────────────────────────┤
      │                                  │   Return SimulationStepInfo    │
      │  Return proxy to                 │                                │
      │  SimulationStepInfo              │                                │
      │◄─────────────────────────────────┤                                │
      │                                  │                                │
      │  obs = result.getObservation()   │                                │
      ├─────────────────────────────────►│                                │
      │                                  │  Call getObservation()         │
      │                                  ├───────────────────────────────►│
      │                                  │◄───────────────────────────────┤
      │  Return ObservationState proxy   │   Return ObservationState      │
      │◄─────────────────────────────────┤                                │
      │                                  │                                │
      │  loads = obs.getVmLoads()        │                                │
      ├─────────────────────────────────►│                                │
      │                                  │  Call getVmLoads()             │
      │                                  ├───────────────────────────────►│
      │                                  │◄───────────────────────────────┤
      │  Return [0.45, 0.78, 0.12]       │   Return double[] array        │
      │◄─────────────────────────────────┤                                │
      │                                  │                                │
```

---

## 5. Key Components Explained

### 5.1 LoadBalancerGateway (Java)

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/LoadBalancerGateway.java`

**Purpose:** This is the **entry point** that Python connects to. It coordinates the entire simulation lifecycle.

**Key Methods:**

```java
public class LoadBalancerGateway {
    private SimulationCore simulationCore;
    private GatewayServer gatewayServer;

    /**
     * Initialise new episode
     * @param seed Random seed for reproducibility
     * @return Initial observation and info
     */
    public SimulationStepInfo reset(long seed) {
        // 1. Create new SimulationCore
        this.simulationCore = new SimulationCore(config, seed);

        // 2. Load workload from CSV file
        simulationCore.loadWorkload(workloadFile);

        // 3. Create infrastructure (hosts, VMs)
        simulationCore.createInfrastructure();

        // 4. Get initial observation
        ObservationState obs = getCurrentState();

        // 5. Return initial state
        return new SimulationStepInfo(obs, 0.0, false, false, infoMap);
    }

    /**
     * Execute one step: assign cloudlet to VM and advance simulation
     * @param vmId Target VM ID (-1 for no-op)
     * @return Observation, reward, terminated, truncated, info
     */
    public SimulationStepInfo step(int vmId) {
        boolean wasInvalidAction = false;

        // 1. Get next pending cloudlet from queue
        Cloudlet nextCloudlet = broker.getNextPendingCloudlet();

        if (vmId < 0) {
            // No-op action: Don't assign, just advance time
            isNoOpAction = true;
        } else {
            // 2. Validate action
            Vm targetVm = getVmById(vmId);
            if (targetVm == null || !canAssignCloudlet(nextCloudlet, targetVm)) {
                wasInvalidAction = true;
                // Still need to assign to keep simulation moving
                targetVm = findAlternativeVm(nextCloudlet);
            }

            // 3. Assign cloudlet to VM
            broker.assignCloudletToVm(nextCloudlet, targetVm);
        }

        // 4. Advance simulation by SCHEDULING_INTERVAL (e.g., 1.0 second)
        simulation.runFor(SCHEDULING_INTERVAL);

        // 5. Check termination conditions
        boolean terminated = broker.allCloudletsFinished();
        boolean truncated = currentClock >= maxEpisodeLength;

        // 6. Calculate episode statistics (only at episode end)
        double episodeDuration = 0;
        int episodeCompletedCloudlets = 0;
        int episodeTotalCloudlets = 0;
        double episodeCompletionRate = 0;

        if (terminated || truncated) {
            episodeDuration = currentClock;
            episodeCompletedCloudlets = broker.getCloudletFinishedList().size();
            episodeTotalCloudlets = broker.getInputCloudlets().size();
            episodeCompletionRate = episodeTotalCloudlets > 0
                ? (double) episodeCompletedCloudlets / episodeTotalCloudlets
                : 0.0;
        }

        // 7. Collect new observation
        ObservationState obs = getCurrentState();

        // 8. Calculate reward
        double reward = calculateReward(wasInvalidAction);

        // 9. Build info dictionary
        Map<String, Object> infoMap = buildInfoMap(reward, episodeDuration,
            episodeCompletedCloudlets, episodeTotalCloudlets, episodeCompletionRate);

        // 10. Return step result
        return new SimulationStepInfo(obs, reward, terminated, truncated, infoMap,
            episodeDuration, episodeCompletedCloudlets, episodeTotalCloudlets,
            episodeCompletionRate);
    }

    /**
     * Collect current system state into observation
     */
    private ObservationState getCurrentState() {
        // Extract VM metrics
        double[] vmLoads = new double[maxVms];
        int[] vmTypes = new int[maxVms];
        int[] vmAvailablePes = new int[maxVms];

        for (int i = 0; i < activeVms.size(); i++) {
            Vm vm = activeVms.get(i);
            vmLoads[i] = vm.getCpuPercentUtilization();
            vmTypes[i] = getVmTypeCode(vm);  // 1=Small, 2=Medium, 3=Large
            vmAvailablePes[i] = (int) vm.getFreePesNumber();
        }

        // Extract queue information
        int waitingCloudlets = broker.getPendingCloudletsCount();
        Cloudlet nextCloudlet = broker.peekNextCloudlet();

        int nextCloudletPes = nextCloudlet != null ? (int) nextCloudlet.getNumberOfPes() : 0;
        long nextCloudletMi = nextCloudlet != null ? nextCloudlet.getLength() : 0;
        double nextCloudletWaitTime = nextCloudlet != null
            ? currentClock - nextCloudlet.getSubmissionDelay()
            : 0.0;

        // Analyze queue composition
        int[] queuePesDistribution = analyzeQueuePesDistribution();  // [small, medium, large]

        // Historical statistics
        int completedLast10 = getCompletedCloudletsLast10Steps();

        return new ObservationState(
            vmLoads, vmTypes, vmAvailablePes,
            waitingCloudlets, nextCloudletPes, nextCloudletMi,
            nextCloudletWaitTime, queuePesDistribution,
            completedLast10,
            activeVms.size()  // actual VM count
        );
    }

    /**
     * Calculate multi-objective reward
     */
    private double calculateReward(boolean wasInvalidAction) {
        // Component 1: Wait time penalty
        double avgWaitTime = calculateAverageWaitTime();
        double rewardWaitTime = -avgWaitTime * config.rewardWaitTimeCoef;

        // Component 2: Unutilization penalty
        double avgUtilization = calculateAverageUtilization();
        double unutilization = 1.0 - avgUtilization;
        double rewardUnutilization = -unutilization * config.rewardUnutilizationCoef;

        // Component 3: Queue penalty
        int queueLength = broker.getPendingCloudletsCount();
        double rewardQueue = -queueLength * config.rewardQueuePenaltyCoef;

        // Component 4: Invalid action penalty
        double rewardInvalidAction = wasInvalidAction
            ? -config.rewardInvalidActionCoef
            : 0.0;

        // Total reward
        double totalReward = rewardWaitTime + rewardUnutilization
                           + rewardQueue + rewardInvalidAction;

        // Store components in info for logging
        lastRewardComponents.put("reward_wait_time", rewardWaitTime);
        lastRewardComponents.put("reward_unutilization", rewardUnutilization);
        lastRewardComponents.put("reward_queue_penalty", rewardQueue);
        lastRewardComponents.put("reward_invalid_action", rewardInvalidAction);

        return totalReward;
    }
}
```

**Lifecycle:**
1. Java `Main.java` starts GatewayServer with LoadBalancerGateway as entry point
2. Python connects and calls `reset(seed)`
3. Python calls `step(action)` repeatedly until done
4. Python calls `reset(seed)` to start next episode

---

### 5.2 SimulationCore (Java)

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/SimulationCore.java`

**Purpose:** Manages the CloudSim Plus simulation instance, creates infrastructure, loads workload.

**Key Responsibilities:**
- Initialize CloudSimPlus `Simulation` object
- Create `Datacenter` with hosts
- Create VMs (Small: 2 PEs, Medium: 4 PEs, Large: 8 PEs)
- Load cloudlets from CSV file
- Provide `simulation.runFor(interval)` to advance time

---

### 5.3 LoadBalancingBroker (Java)

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/LoadBalancingBroker.java`

**Purpose:** Custom CloudSim broker that **waits for agent decisions** instead of auto-scheduling.

**Key Feature:**
```java
public class LoadBalancingBroker extends DatacenterBrokerSimple {
    private PriorityQueue<Cloudlet> pendingCloudlets;  // Ordered by arrival time

    /**
     * Override default scheduling: Don't auto-assign, wait for agent
     */
    @Override
    protected void requestDatacenterToCreateWaitingCloudlets() {
        // DO NOTHING - agent will explicitly assign cloudlets
    }

    /**
     * Agent calls this to assign specific cloudlet to specific VM
     */
    public void assignCloudletToVm(Cloudlet cloudlet, Vm vm) {
        cloudlet.setVm(vm);
        submitCloudletToVm(cloudlet, vm);
    }

    /**
     * Peek at next cloudlet without removing from queue
     */
    public Cloudlet peekNextCloudlet() {
        return pendingCloudlets.peek();
    }
}
```

---

### 5.4 ObservationState (Java)

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/ObservationState.java`

**Purpose:** Data container for observations transferred to Python.

```java
public class ObservationState {
    // VM metrics (padded to maxVms)
    private final double[] vmLoads;           // CPU utilization [0.0-1.0]
    private final int[] vmTypes;              // 1=Small, 2=Medium, 3=Large
    private final int[] vmAvailablePes;       // Free PEs per VM

    // Queue information
    private final int waitingCloudlets;       // Number of tasks in queue
    private final int nextCloudletPes;        // PEs required by next task
    private final long nextCloudletMi;        // Workload (MI) of next task
    private final double nextCloudletWaitTime; // How long next task has waited
    private final int[] queuePesDistribution; // [small, medium, large] task counts

    // Historical statistics
    private final int completedCloudletsLast10Steps;  // Throughput indicator

    // Metadata
    private final int actualVmCount;          // Actual number of VMs (rest is padding)

    // Getters for all fields (Py4J requires getters)
    public double[] getVmLoads() { return vmLoads; }
    public int[] getVmTypes() { return vmTypes; }
    // ... etc
}
```

**Why Padding?**
- Gym spaces must have fixed dimensions
- But number of VMs can vary (3-50 depending on config)
- Solution: Pad arrays to `maxVms` (e.g., 50), track `actualVmCount`

---

### 5.5 SimulationStepInfo (Java)

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/SimulationStepInfo.java`

**Purpose:** Complete step result returned to Python (obs, reward, done, info).

```java
public class SimulationStepInfo {
    private final ObservationState observation;
    private final double reward;
    private final boolean terminated;  // Episode ended naturally (all tasks done)
    private final boolean truncated;   // Episode cut off (time limit reached)
    private final Map<String, Object> info;

    // Episode statistics (only populated at episode end)
    private final double episodeDuration;          // Total episode time (seconds)
    private final int episodeCompletedCloudlets;   // Number of completed tasks
    private final int episodeTotalCloudlets;       // Total tasks in episode
    private final double episodeCompletionRate;    // Completion ratio

    // Getters
    public ObservationState getObservation() { return observation; }
    public double getReward() { return reward; }
    public boolean isTerminated() { return terminated; }
    public boolean isTruncated() { return truncated; }
    public Map<String, Object> getInfo() { return info; }

    /**
     * Convert info map to format suitable for Python
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>(info);
        map.put("episode_duration", episodeDuration);
        map.put("episode_completed_cloudlets", episodeCompletedCloudlets);
        map.put("episode_total_cloudlets", episodeTotalCloudlets);
        map.put("episode_completion_rate", episodeCompletionRate);
        return map;
    }
}
```

---

### 5.6 LoadBalancingEnv (Python)

**Location:** `drl-manager/gym_cloudsimplus/envs/loadbalancing_env.py`

**Purpose:** Gymnasium environment that wraps the Java simulation.

```python
class LoadBalancingEnv(gym.Env):
    """
    OpenAI Gym environment for cloud load balancing
    """

    def __init__(self, config):
        super(LoadBalancingEnv, self).__init__()

        self.config = config
        self.num_vms = config["initial_s_vm_count"] + \
                       config["initial_m_vm_count"] + \
                       config["initial_l_vm_count"]

        # Connect to Java via Py4J
        self.gateway = JavaGateway(
            gateway_parameters=GatewayParameters(auto_convert=True)
        )
        self.loadbalancer_gateway = self.gateway.entry_point

        # Define action space: [0, 1, 2, ..., num_vms]
        # 0 = No-op, 1 = VM 0, 2 = VM 1, etc.
        self.action_space = spaces.Discrete(self.num_vms + 1)

        # Define observation space (Dict of multiple arrays)
        self.observation_space = spaces.Dict({
            "vm_loads": spaces.Box(low=0.0, high=1.0,
                shape=(self.num_vms,), dtype=np.float32),
            "vm_available_pes": spaces.Box(low=0, high=8,
                shape=(self.num_vms,), dtype=np.int32),
            "vm_types": spaces.Box(low=0, high=3,
                shape=(self.num_vms,), dtype=np.int32),
            "waiting_cloudlets": spaces.Box(low=0, high=np.inf,
                shape=(1,), dtype=np.float32),
            "next_cloudlet_pes": spaces.Box(low=0, high=np.inf,
                shape=(1,), dtype=np.float32),
            "next_cloudlet_mi": spaces.Box(low=0, high=np.inf,
                shape=(1,), dtype=np.float32),
            "next_cloudlet_wait_time": spaces.Box(low=0, high=np.inf,
                shape=(1,), dtype=np.float32),
            "queue_pes_distribution": spaces.Box(low=0, high=np.inf,
                shape=(3,), dtype=np.int32),
            "completed_cloudlets_last_10_steps": spaces.Box(low=0, high=np.inf,
                shape=(1,), dtype=np.int32)
        })

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode
        """
        super(LoadBalancingEnv, self).reset(seed=seed, options=options)

        current_seed = seed if seed is not None else self.config.get("seed", 42)

        try:
            # Call Java reset
            reset_result_java = self.loadbalancer_gateway.reset(current_seed)

            # Extract observation
            observation = self._get_obs(reset_result_java.getObservation())

            # Extract info
            info = self._process_info(reset_result_java.getInfo())
            info["actual_vm_count"] = reset_result_java.getObservation().getActualVmCount()

            return (observation, info)

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            # Return dummy observation
            return (self._get_dummy_obs(), {"error": "Reset failed"})

    def step(self, action: int):
        """
        Execute one step: agent chooses VM, simulation advances
        """
        self.current_step += 1

        # Map action to VM ID
        target_vm_id = int(action) - 1  # 0→-1, 1→0, 2→1, etc.

        try:
            # Call Java step
            step_result_java = self.loadbalancer_gateway.step(target_vm_id)

            # Extract results
            observation = self._get_obs(step_result_java.getObservation())
            reward = float(step_result_java.getReward())
            terminated = bool(step_result_java.isTerminated())
            truncated = bool(step_result_java.isTruncated())
            info = self._process_info(step_result_java.getInfo())

            # Accumulate episode statistics
            self._ep_steps += 1
            for k in ["reward_wait_time", "reward_unutilization",
                      "reward_queue_penalty", "reward_invalid_action"]:
                if k in info:
                    self._ep_sum[k] += float(info[k])

            # On episode end, add episode-level averages
            if terminated or truncated:
                steps = max(1, self._ep_steps)
                info["episode_reward_wait_time_mean"] = self._ep_sum["reward_wait_time"] / steps
                info["episode_reward_unutilization_mean"] = self._ep_sum["reward_unutilization"] / steps
                info["episode_reward_queue_penalty_mean"] = self._ep_sum["reward_queue_penalty"] / steps
                info["episode_reward_invalid_action_mean"] = self._ep_sum["reward_invalid_action"] / steps

                # Reset accumulators
                self._ep_steps = 0
                for k in self._ep_sum:
                    self._ep_sum[k] = 0.0

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Step failed: {e}")
            return (self._get_dummy_obs(), 0.0, True, False, {"error": "Step failed"})

    def _get_obs(self, java_obs_state):
        """
        Convert Java ObservationState to Python dict
        """
        # Extract arrays (Py4J auto-converts to lists)
        vm_loads_list = java_obs_state.getVmLoads()
        vm_types_list = java_obs_state.getVmTypes()
        vm_available_pes_list = java_obs_state.getVmAvailablePes()
        queue_pes_dist_list = java_obs_state.getQueuePesDistribution()

        # Convert to NumPy, slice to actual VM count
        vm_loads = np.array(vm_loads_list[:self.num_vms], dtype=np.float32)
        vm_types = np.array(vm_types_list[:self.num_vms], dtype=np.int32)
        vm_available_pes = np.array(vm_available_pes_list[:self.num_vms], dtype=np.int32)
        queue_pes_dist = np.array(queue_pes_dist_list, dtype=np.int32)

        # Extract scalars
        waiting_cloudlets = np.array([java_obs_state.getWaitingCloudlets()], dtype=np.float32)
        next_cloudlet_pes = np.array([java_obs_state.getNextCloudletPes()], dtype=np.float32)
        next_cloudlet_mi = np.array([float(java_obs_state.getNextCloudletMi())], dtype=np.float32)
        next_cloudlet_wait_time = np.array([java_obs_state.getNextCloudletWaitTime()], dtype=np.float32)
        completed_last_10 = np.array([java_obs_state.getCompletedCloudletsLast10Steps()], dtype=np.int32)

        # Build observation dictionary
        obs_dict = {
            "vm_loads": vm_loads,
            "vm_available_pes": vm_available_pes,
            "vm_types": vm_types,
            "waiting_cloudlets": waiting_cloudlets,
            "next_cloudlet_pes": next_cloudlet_pes,
            "next_cloudlet_mi": next_cloudlet_mi,
            "next_cloudlet_wait_time": next_cloudlet_wait_time,
            "queue_pes_distribution": queue_pes_dist,
            "completed_cloudlets_last_10_steps": completed_last_10
        }

        return obs_dict

    def _process_info(self, java_info_map):
        """
        Convert Java Map<String, Object> to Python dict
        Py4J auto-converts Map to dict
        """
        return dict(java_info_map)
```

---

### 5.7 Training Script (Python)

**Location:** `drl-manager/mnt/train.py`

**Purpose:** Main script that loads config, creates environment, trains agent.

```python
import sys
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from gym_cloudsimplus.envs.loadbalancing_env import LoadBalancingEnv

def load_config(config_path, experiment_name):
    """Load YAML config and merge common + experiment-specific"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    common = config.get("common", {})
    experiment = config.get(experiment_name, {})

    # Merge (experiment overrides common)
    merged = {**common, **experiment}
    return merged

def main():
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "experiment_11_long"

    # Load config
    config = load_config("../config.yml", experiment_name)

    # Create log directory
    log_dir = Path(f"../logs/{config['experiment_type_dir']}/{config['experiment_name']}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define which info keys to log to CSV
    info_keywords_to_log = (
        "reward_wait_time",
        "reward_unutilization",
        "reward_queue_penalty",
        "reward_invalid_action",
        "episode_duration",
        "episode_completed_cloudlets",
        "episode_total_cloudlets",
        "episode_completion_rate"
    )

    # Create environment
    env = LoadBalancingEnv(config)
    env = Monitor(env, str(log_dir), info_keywords=info_keywords_to_log)

    # Select algorithm
    algorithm = config.get("algorithm", "PPO")

    if algorithm == "RecurrentPPO":
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            policy_kwargs=config.get("policy_kwargs", {}),
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard")
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            # ... similar params
        )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="model"
    )
    callbacks = CallbackList([checkpoint_callback])

    # Train
    print(f"Starting training for {config['timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["timesteps"],
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    model.save(str(log_dir / "final_model"))
    print(f"Training complete. Model saved to {log_dir}")

if __name__ == "__main__":
    main()
```

---

## 6. Observation and Action Spaces

### 6.1 Observation Space

The observation is a **dictionary** with 9 keys:

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `vm_loads` | float32 | (N,) | [0.0, 1.0] | CPU utilization per VM |
| `vm_available_pes` | int32 | (N,) | [0, 8] | Free PEs per VM |
| `vm_types` | int32 | (N,) | [0, 3] | VM type: 0=padding, 1=Small, 2=Medium, 3=Large |
| `waiting_cloudlets` | float32 | (1,) | [0, ∞) | Number of tasks in queue |
| `next_cloudlet_pes` | float32 | (1,) | [0, 8] | PEs required by next task |
| `next_cloudlet_mi` | float32 | (1,) | [0, ∞) | Workload (MI) of next task |
| `next_cloudlet_wait_time` | float32 | (1,) | [0, ∞) | Wait time of next task (seconds) |
| `queue_pes_distribution` | int32 | (3,) | [0, ∞) | [small_tasks, medium_tasks, large_tasks] |
| `completed_cloudlets_last_10_steps` | int32 | (1,) | [0, ∞) | Throughput in last 10 steps |

**Total Dimensions** (for 3 VMs): 3 + 3 + 3 + 1 + 1 + 1 + 1 + 3 + 1 = **17 values**

**Example Observation at Timestep 100:**
```python
{
    "vm_loads": [0.45, 0.78, 0.12],           # VM 0: 45%, VM 1: 78%, VM 2: 12%
    "vm_available_pes": [1, 0, 7],            # VM 0: 1 free PE, VM 1: full, VM 2: 7 free
    "vm_types": [1, 2, 3],                    # Small, Medium, Large
    "waiting_cloudlets": [5.0],               # 5 tasks waiting
    "next_cloudlet_pes": [3.0],               # Next task needs 3 PEs
    "next_cloudlet_mi": [7234567.0],          # Next task workload
    "next_cloudlet_wait_time": [12.5],        # Next task waited 12.5 seconds
    "queue_pes_distribution": [2, 2, 1],      # 2 small, 2 medium, 1 large in queue
    "completed_cloudlets_last_10_steps": [8]  # Completed 8 tasks in last 10 steps
}
```

---

### 6.2 Action Space

**Type:** `Discrete(N+1)` where N = number of VMs

**Action Mapping:**

- `0`: **No-op** (don't assign next task, wait for next step)
- `1`: Assign next task to **VM 0**
- `2`: Assign next task to **VM 1**
- `3`: Assign next task to **VM 2**
- ...
- `N`: Assign next task to **VM (N-1)**

**Example (3 VMs):**
- Action space size: 4
- Valid actions: {0, 1, 2, 3}

**Invalid Actions:**

An action is considered invalid if:
1. Target VM doesn't have enough free PEs
2. Target VM doesn't exist (rare, only if VM count changes)

**What happens on invalid action?**
- Penalty: Reward component `reward_invalid_action` is large negative value
- Fallback: Java finds an alternative VM to keep simulation running
- Info: `info["was_invalid_action"] = True`

---

## 7. Reward Function Design

The reward function is **multi-objective**, combining 4 components:

### 7.1 Component 1: Wait Time Penalty

**Goal:** Minimize task waiting times

**Formula:**
```
reward_wait_time = -avg_wait_time × weight_wait_time
```

**Details:**
- `avg_wait_time`: Average of all waiting cloudlets' wait times (seconds)
- `weight_wait_time`: Configurable coefficient (typically 1.0)

**Example:**
- 5 tasks waiting: [10s, 15s, 8s, 20s, 12s]
- avg_wait_time = (10+15+8+20+12)/5 = 13 seconds
- reward_wait_time = -13 × 1.0 = **-13.0**

---

### 7.2 Component 2: Unutilization Penalty

**Goal:** Maximize resource utilization

**Formula:**
```
unutilization = 1.0 - avg_vm_utilization
reward_unutilization = -unutilization × weight_unutilization
```

**Details:**

- `avg_vm_utilization`: Average CPU utilization across all active VMs
- `weight_unutilization`: Configurable coefficient (typically 0.5)

**Example:**
- 3 VMs with utilization: [0.45, 0.78, 0.12]
- avg_utilization = (0.45+0.78+0.12)/3 = 0.45 (45%)
- unutilization = 1.0 - 0.45 = 0.55
- reward_unutilization = -0.55 × 0.5 = **-0.275**

---

### 7.3 Component 3: Queue Penalty

**Goal:** Keep queue short

**Formula:**
```
reward_queue_penalty = -queue_length × weight_queue_penalty
```

**Details:**
- `queue_length`: Number of waiting cloudlets
- `weight_queue_penalty`: Configurable coefficient (typically 0.8)

**Example:**
- queue_length = 5
- reward_queue_penalty = -5 × 0.8 = **-4.0**

---

### 7.4 Component 4: Invalid Action Penalty

**Goal:** Discourage invalid actions

**Formula:**
```
reward_invalid_action = -weight_invalid_action  (if action was invalid)
                      = 0.0                      (otherwise)
```

**Details:**
- `weight_invalid_action`: Large coefficient (typically 2.0)

**Example:**
- If agent tried to assign 4-PE task to VM with 2 free PEs:
- reward_invalid_action = **-2.0**

---

### 7.5 Total Reward

```
total_reward = reward_wait_time
             + reward_unutilization
             + reward_queue_penalty
             + reward_invalid_action
```

**Example Calculation:**
```
reward_wait_time = -13.0
reward_unutilization = -0.275
reward_queue_penalty = -4.0
reward_invalid_action = 0.0

total_reward = -13.0 - 0.275 - 4.0 + 0.0 = -17.275
```

**Typical Range:** -50 to 0 (negative, agent tries to maximize toward 0)

---

### 7.6 Reward Weight Tuning

Adjust weights in `config.yml` to change optimization priorities:

```yaml
# Emphasize wait time reduction
reward_wait_time_coef: 2.0       # Double importance
reward_unutilization_coef: 0.3   # Less important
reward_queue_penalty_coef: 0.5
reward_invalid_action_coef: 5.0  # Strongly discourage

# Emphasize utilization
reward_wait_time_coef: 0.5
reward_unutilization_coef: 2.0   # Double importance
reward_queue_penalty_coef: 0.5
reward_invalid_action_coef: 2.0

# Balanced (default)
reward_wait_time_coef: 1.0
reward_unutilization_coef: 0.5
reward_queue_penalty_coef: 0.8
reward_invalid_action_coef: 2.0
```

---

## 8. Complete Workflow

### 8.1 System Startup

**Terminal 1: Start Java Gateway**
```bash
cd cloudsimplus-gateway
./gradlew run

# Output:
# Starting server: 0.0.0.0 25333
# GatewayServer started, listening on port 25333
```

Java process now waits for Python connection.

**Terminal 2: Start Python Training**
```bash
cd drl-manager
.venv/Scripts/Activate.ps1
python mnt/entrypoint.py --experiment experiment_11_long

# Output:
# Python Env connected!
# Starting training for 500000 timesteps...
```

### 8.2 Episode Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                         Episode Start                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Python: reset() │
                    └──────────────────┘
                              │
                              ▼ Py4J call
                    ┌──────────────────┐
                    │  Java: reset()   │
                    │  • Load workload │
                    │  • Create VMs    │
                    │  • Initial obs   │
                    └──────────────────┘
                              │
                              ▼ Return initial_obs
┌─────────────────────────────────────────────────────────────────┐
│                         Step Loop (Repeat)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌────────────────────────────────────────────────────┐        │
│   │ Python: Agent observes state                       │        │
│   │ obs = { vm_loads, queue_info, next_cloudlet, ... }│        │
│   └────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼                                   │
│   ┌────────────────────────────────────────────────────┐        │
│   │ Python: Agent selects action                       │        │
│   │ action = model.predict(obs)                        │        │
│   │ # e.g., action = 2 (assign to VM 1)                │        │
│   └────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼ Py4J call                         │
│   ┌────────────────────────────────────────────────────┐        │
│   │ Java: step(action)                                 │        │
│   │ • Assign cloudlet to selected VM                   │        │
│   │ • Advance simulation by 1.0 second                 │        │
│   │ • Calculate reward components                      │        │
│   │ • Collect new observation                          │        │
│   │ • Check if done (all tasks finished OR time limit)│        │
│   └────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼ Return (obs, reward, done, info)  │
│   ┌────────────────────────────────────────────────────┐        │
│   │ Python: Store transition in replay buffer          │        │
│   │ (obs, action, reward, next_obs, done)              │        │
│   └────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                          │
│                    │   If done?       │                          │
│                    └──────────────────┘                          │
│                       │           │                              │
│                      No          Yes                             │
│                       │           │                              │
│                       └───────┐   │                              │
│                               │   │                              │
└───────────────────────────────┘   │                              │
                                    ▼
                          ┌──────────────────┐
                          │ Episode Complete │
                          │ • Log metrics    │
                          │ • Update policy  │
                          │ • Start new ep.  │
                          └──────────────────┘
```

### 8.3 Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                  At Each Step (every ~1 second)                 │
└────────────────────────────────────────────────────────────────┘

Python Side                           Java Side
────────────────────────────────      ─────────────────────────────

1. Agent Policy Network               4. LoadBalancerGateway.step()
   ┌──────────────────┐                  ┌──────────────────────┐
   │ LSTM + MLP       │                  │ • Get action (vmId)  │
   │ Input: obs_t     │                  │ • Validate action    │
   │ Output: action   │                  │ • Assign cloudlet    │
   └──────────────────┘                  └──────────────────────┘
          │                                        │
          │ action = 2                             │
          ▼                                        ▼
2. Environment.step(2)                  5. Simulation.runFor(1.0)
   ┌──────────────────┐                  ┌──────────────────────┐
   │ Map action:      │                  │ CloudSim advances:   │
   │ 2 → vmId=1       │                  │ • Process events     │
   │                  │                  │ • Execute cloudlets  │
   └──────────────────┘                  │ • Update metrics     │
          │                               └──────────────────────┘
          │ Py4J call                               │
          ▼                                         ▼
3. gateway.entry_point.step(1)          6. Build SimulationStepInfo
   ┌──────────────────┐                  ┌──────────────────────┐
   │ TCP to Java      │◄─────────────────┤ obs = getCurrentState│
   │ localhost:25333  │                  │ reward = calculate() │
   └──────────────────┘                  │ done = checkDone()   │
          ▲                               └──────────────────────┘
          │                                        │
          │ Return via Py4J                        │ Return object
          │                                        │
7. Unpack result                                   │
   ┌──────────────────┐                            │
   │ obs = result.getObservation()                 │
   │ reward = result.getReward()  ◄────────────────┘
   │ done = result.isTerminated()
   │ info = result.getInfo()
   └──────────────────┘
          │
          ▼
8. Store in replay buffer
   ┌──────────────────┐
   │ (s_t, a_t, r_t,  │
   │  s_{t+1}, done)  │
   └──────────────────┘
          │
          ▼
9. Update policy (PPO)
   ┌──────────────────┐
   │ Every N steps:   │
   │ • Compute returns│
   │ • Compute advant.│
   │ • Update network │
   └──────────────────┘
```

---

## 9. Training Process

### 9.1 Training Loop Pseudocode

```python
# Initialize
model = RecurrentPPO(policy="MultiInputLstmPolicy", env=env, ...)
total_timesteps = 500000
n_steps = 2048  # Collect 2048 steps before each update

# Training loop
for update in range(total_timesteps // n_steps):
    # 1. Collect n_steps of experience
    rollout_buffer = []
    obs = env.reset()

    for step in range(n_steps):
        # Agent selects action (with exploration)
        action, lstm_state = model.predict(obs, lstm_state)

        # Environment executes action
        next_obs, reward, done, info = env.step(action)

        # Store transition
        rollout_buffer.append((obs, action, reward, done, lstm_state))

        obs = next_obs
        if done:
            obs = env.reset()

    # 2. Compute returns and advantages
    returns = compute_returns(rollout_buffer, gamma=0.995)
    advantages = compute_gae(rollout_buffer, gamma=0.995, gae_lambda=0.95)

    # 3. Update policy (multiple epochs on same data)
    for epoch in range(n_epochs):
        for batch in minibatches(rollout_buffer, batch_size=256):
            # Compute PPO loss
            policy_loss = ppo_policy_loss(batch, advantages, clip_range=0.2)
            value_loss = mse_loss(batch, returns)
            entropy_loss = entropy(policy)

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    # 4. Log metrics
    logger.log("train/policy_loss", policy_loss)
    logger.log("train/value_loss", value_loss)
    logger.log("rollout/ep_rew_mean", mean_episode_reward)
```

### 9.2 LSTM Policy Architecture

For `MultiInputLstmPolicy`:

```
Input Observation Dict
  ├─ vm_loads: [0.45, 0.78, 0.12]
  ├─ vm_types: [1, 2, 3]
  ├─ vm_available_pes: [1, 0, 7]
  ├─ waiting_cloudlets: [5.0]
  ├─ next_cloudlet_pes: [3.0]
  ├─ next_cloudlet_mi: [7234567.0]
  ├─ next_cloudlet_wait_time: [12.5]
  ├─ queue_pes_distribution: [2, 2, 1]
  └─ completed_cloudlets_last_10_steps: [8]

          ↓ (Flatten & Concatenate)

[0.45, 0.78, 0.12, 1, 2, 3, 1, 0, 7, 5.0, 3.0, 7234567.0, 12.5, 2, 2, 1, 8]
                        ↓
              ┌─────────────────┐
              │ Feature Extractor│
              │ (Linear Layer)   │
              │ Input: 17        │
              │ Output: 256      │
              └─────────────────┘
                        ↓
              ┌─────────────────┐
              │ LSTM Layer       │
              │ Hidden: 256      │
              │ Layers: 1        │
              │ (maintains state)│
              └─────────────────┘
                        ↓
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌────────────────┐           ┌────────────────┐
│ Policy Head    │           │ Value Head     │
│ MLP [128, 128] │           │ MLP [128, 128] │
│ Output: 4      │           │ Output: 1      │
│ (action logits)│           │ (state value)  │
└────────────────┘           └────────────────┘
         │                             │
         ▼                             ▼
   Action Probs               Value Estimate
   [0.1, 0.6, 0.25, 0.05]     V(s) = -15.3
```

### 9.3 Training Metrics

**Logged to TensorBoard:**
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length (steps)
- `train/policy_loss`: PPO policy loss
- `train/value_loss`: Value function MSE loss
- `train/entropy_loss`: Policy entropy (exploration)
- `train/clip_fraction`: Fraction of clipped updates
- `train/learning_rate`: Current learning rate

**Logged to CSV (Monitor):**
- Episode reward
- Episode length
- Episode time
- Custom info keys:
  - `reward_wait_time`
  - `reward_unutilization`
  - `reward_queue_penalty`
  - `reward_invalid_action`
  - `episode_duration`
  - `episode_completed_cloudlets`
  - `episode_total_cloudlets`
  - `episode_completion_rate`

---

## 10. Project Structure

```
rl-cloudsimplus-greenscheduling/
│
├── cloudsimplus-gateway/               # Java simulation
│   ├── src/main/java/giu/edu/cspg/
│   │   ├── Main.java                   # Py4J server entry point
│   │   ├── LoadBalancerGateway.java    # Entry point for Python
│   │   ├── SimulationCore.java         # CloudSim wrapper
│   │   ├── LoadBalancingBroker.java    # Custom broker
│   │   ├── ObservationState.java       # Observation data class
│   │   ├── SimulationStepInfo.java     # Step result data class
│   │   └── utils/
│   │       └── WorkloadFileReader.java # CSV/SWF parser
│   ├── src/main/resources/
│   │   └── traces/                     # Workload CSV files
│   │       ├── poisson_04_3600.csv
│   │       └── three_60max_8maxcores.csv
│   ├── build.gradle                    # Gradle build config
│   └── gradlew                         # Gradle wrapper
│
├── drl-manager/                        # Python RL training
│   ├── gym_cloudsimplus/
│   │   └── envs/
│   │       └── loadbalancing_env.py    # Gym environment
│   ├── mnt/
│   │   └── train.py                    # Main training script
│   ├── generate_workload.py            # Workload generator
│   ├── requirements.txt                # Python dependencies
│   └── .venv/                          # Python virtual environment
│
├── config.yml                          # Central configuration
├── logs/                               # Training logs & models
│   └── LongEpisode/
│       └── exp11_3600s/
│           ├── final_model.zip
│           ├── monitor.csv
│           └── tensorboard/
│
├── docs/                               # Documentation
│   ├── SYSTEM_ARCHITECTURE_AND_PY4J_COMMUNICATION.md  # This file
│   └── ...
│
└── README.md                           # Project overview
```

---

## 11. Configuration System

### 11.1 YAML Structure

`config.yml` has two main sections:

1. **common**: Shared defaults for all experiments
2. **experiment_X**: Experiment-specific overrides

```yaml
common:
  # Infrastructure defaults
  hosts_count: 8
  host_pes: 16
  initial_s_vm_count: 6
  initial_m_vm_count: 3
  initial_l_vm_count: 2

  # RL algorithm defaults
  algorithm: "PPO"
  learning_rate: 0.0003
  gamma: 0.99
  # ...

experiment_11_long:
  # Override specific values
  simulation_name: "Exp11_LongEpisode_3600s"
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/poisson_04_3600.csv"
  max_episode_length: 3600

  hosts_count: 16      # Override common.hosts_count
  initial_s_vm_count: 20
  initial_m_vm_count: 10
  initial_l_vm_count: 5

  algorithm: "RecurrentPPO"  # Override common.algorithm
  timesteps: 500000
  # ...
```

### 11.2 Key Configuration Categories

**Infrastructure:**
- `hosts_count`, `host_pes`: Host resources
- `initial_s_vm_count`, `initial_m_vm_count`, `initial_l_vm_count`: Initial VMs
- `small_vm_pes`, `medium_vm_pes`, `large_vm_pes`: VM sizes (2, 4, 8 PEs)

**Workload:**
- `workload_mode`: "CSV" or "Random"
- `cloudlet_trace_file`: Path to CSV file (relative to traces/)
- `max_episode_length`: Episode timeout (seconds)

**RL Algorithm:**
- `algorithm`: "PPO", "RecurrentPPO", "MaskablePPO"
- `policy`: "MultiInputPolicy" or "MultiInputLstmPolicy"
- `timesteps`: Total training steps
- `learning_rate`, `gamma`, `gae_lambda`: RL hyperparameters
- `n_steps`, `batch_size`, `n_epochs`: PPO-specific params

**Policy Network:**
- `policy_kwargs`: Architecture details
  - `lstm_hidden_size`: LSTM hidden state size
  - `n_lstm_layers`: Number of LSTM layers
  - `net_arch`: MLP layer sizes for policy/value heads

**Reward Weights:**
- `reward_wait_time_coef`
- `reward_unutilization_coef`
- `reward_queue_penalty_coef`
- `reward_invalid_action_coef`

### 11.3 Loading Configuration in Code

**Python:**
```python
import yaml

def load_config(config_path, experiment_name):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    common = config.get("common", {})
    experiment = config.get(experiment_name, {})

    # Merge: experiment overrides common
    merged = {**common, **experiment}
    return merged

config = load_config("../config.yml", "experiment_11_long")
```

**Java:**

Java reads config via system properties or environment variables passed by Python, or hardcoded defaults.

---

## 12. How to Run

### 12.1 Prerequisites

**System Requirements:**
- Java 17+
- Python 3.10+
- 8GB+ RAM (for simulation)
- 16GB+ RAM recommended for long episodes

**Install Java Dependencies:**
```bash
cd cloudsimplus-gateway
./gradlew build
```

**Install Python Dependencies:**
```bash
cd drl-manager
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 12.2 Running an Experiment

**Step 1: Start Java Gateway (Terminal 1)**
```bash
cd cloudsimplus-gateway
./gradlew run

# Wait for: "GatewayServer started, listening on port 25333"
```

**Step 2: Start Training (Terminal 2)**
```bash
cd drl-manager
source .venv/bin/activate  # Activate virtual environment

python mnt/train.py --experiment experiment_11_long

# Training will start, showing progress bar
# Logs saved to: logs/LongEpisode/exp11_3600s/
```

**Step 3: Monitor Training (Terminal 3, optional)**
```bash
cd drl-manager
tensorboard --logdir=../logs/LongEpisode/exp11_3600s/tensorboard

# Open browser: http://localhost:6006
```

### 12.3 Generating Custom Workload

```bash
cd drl-manager

python generate_workload.py \
  --type poisson \
  --arrival-rate 0.4 \
  --duration 3600 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/my_workload.csv \
  --seed 42

# Update config.yml:
# cloudlet_trace_file: "traces/my_workload.csv"
```

### 12.4 Evaluating Trained Model

```python
# evaluate.py
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gym_cloudsimplus.envs.loadbalancing_env import LoadBalancingEnv
import yaml

# Load config
with open("../config.yml") as f:
    config = yaml.safe_load(f)
    config = {**config["common"], **config["experiment_11_long"]}

# Create environment
env = LoadBalancingEnv(config)

# Load trained model
model = RecurrentPPO.load("../logs/LongEpisode/exp11_3600s/final_model.zip")

# Run evaluation episodes
num_episodes = 10
episode_rewards = []

for ep in range(num_episodes):
    obs = env.reset()
    lstm_states = None
    done = False
    ep_reward = 0

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward

    episode_rewards.append(ep_reward)
    print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, "
          f"Completion Rate = {info['episode_completion_rate']:.2%}")

print(f"\nAverage Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
```

---

## 13. Hierarchical Multi-Datacenter System

### 13.1 Overview

The system supports **Hierarchical Multi-Agent Reinforcement Learning (MARL)** for managing multiple geographically distributed datacenters. This enables:

- **Global-level decisions**: Intelligent routing of incoming tasks to datacenters based on green energy availability, current load, and datacenter capacity
- **Local-level decisions**: Fine-grained task-to-VM scheduling within each datacenter
- **Coordinated optimization**: Balancing objectives across both levels (energy efficiency globally, performance locally)

### 13.2 Architecture Comparison

**Single-Datacenter Mode (Chapters 1-12):**
```
Python Agent ──Py4J──> LoadBalancerGateway ──> Single Datacenter
                                                  ├─ Hosts
                                                  ├─ VMs
                                                  └─ Cloudlets
```

**Multi-Datacenter Hierarchical Mode:**
```
┌─────────────────────────────────────────────────────────────────┐
│                       Python Side (MARL)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────┐         ┌────────────────────┐          │
│  │  Global Agent      │         │  Local Agents      │          │
│  │  (Routes tasks     │         │  (Schedule tasks   │          │
│  │   to datacenters)  │         │   to VMs in DC)    │          │
│  └────────────────────┘         └────────────────────┘          │
│            │                              │                      │
│            │ Global Actions               │ Local Actions       │
│            │ [DC0, DC1, DC0, ...]         │ {0: VM2, 1: VM5}    │
│            └──────────────┬───────────────┘                      │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │ Py4J (Port 25333)
┌───────────────────────────┼──────────────────────────────────────┐
│                  Java Side│(CloudSim Plus)                       │
├───────────────────────────┼──────────────────────────────────────┤
│                           ▼                                      │
│              ┌──────────────────────────┐                        │
│              │ HierarchicalMultiDCGateway│                       │
│              └──────────────────────────┘                        │
│                           │                                      │
│          ┌────────────────┼────────────────┐                    │
│          │                                  │                    │
│          ▼                                  ▼                    │
│  ┌──────────────────┐           ┌──────────────────┐            │
│  │  GlobalBroker    │           │  MultiDCSimCore  │            │
│  │  (Routes tasks)  │           │                  │            │
│  └──────────────────┘           └──────────────────┘            │
│          │                                  │                    │
│          │ Routes to DC                     │ Manages            │
│          ▼                                  ▼                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │           DatacenterInstance 0                      │         │
│  │  ├─ LocalBroker (assigns tasks to VMs)            │         │
│  │  ├─ Hosts [H0, H1, ..., H15]                      │         │
│  │  ├─ VMs [VM0, VM1, ..., VM19]                     │         │
│  │  └─ GreenEnergyProvider (Wind turbine)            │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │           DatacenterInstance 1                      │         │
│  │  ├─ LocalBroker                                    │         │
│  │  ├─ Hosts [H16, H17, ..., H31]                    │         │
│  │  ├─ VMs [VM20, VM21, ..., VM39]                   │         │
│  │  └─ GreenEnergyProvider (Different turbine)       │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │           DatacenterInstance N...                   │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 13.3 Two-Level Decision Making

**Level 1: Global Routing (Global Agent)**

**When:** Tasks arrive at the system (e.g., 5 tasks arrive in current timestep)

**Decision:** Which datacenter should handle each task?

**Observation (Global):**
```python
{
    "dc_green_power": [2500.0, 3200.0, 1800.0],      # kW available per DC
    "dc_queue_sizes": [12.0, 8.0, 15.0],             # Waiting tasks per DC
    "dc_utilizations": [0.65, 0.72, 0.58],           # CPU usage per DC
    "dc_available_pes": [45.0, 38.0, 52.0],          # Free cores per DC
    "upcoming_cloudlets_count": 5,                    # Tasks arriving now
    "next_cloudlet_pes": 4                            # Next task needs 4 cores
}
```

**Action (Global):**
```python
global_actions = [0, 1, 0, 2, 1]  # Route 5 arriving tasks
# Task 0 → DC 0, Task 1 → DC 1, Task 2 → DC 0, Task 3 → DC 2, Task 4 → DC 1
```

**Reward (Global):**
- Green energy ratio (maximize renewable energy usage)
- Total energy consumption (minimize)
- Load balance across datacenters (avoid overloading one DC)
- Total queue sizes across all DCs

---

**Level 2: Local Scheduling (Local Agents, one per DC)**

**When:** Each datacenter has tasks in its local queue

**Decision:** Which VM should handle the next task in this DC's queue?

**Observation (Local, per DC):**
```python
local_observations = {
    0: {  # Datacenter 0
        "host_loads": [0.62, 0.58, 0.71, 0.65, ...],   # 16 hosts
        "vm_loads": [0.45, 0.78, 0.12, ...],           # VM utilizations
        "vm_types": [1, 2, 3, 1, 2, ...],              # Small/Medium/Large
        "vm_available_pes": [1, 0, 7, 2, ...],         # Free cores per VM
        "waiting_cloudlets": 12,                        # Local queue size
        "next_cloudlet_pes": 3                          # Next task needs 3 cores
    },
    1: {  # Datacenter 1
        "host_loads": [...],
        # ... similar structure
    },
    2: { ... }
}
```

**Action (Local):**
```python
local_actions = {
    0: 5,   # DC 0: Assign next task to VM 5
    1: 12,  # DC 1: Assign next task to VM 12
    2: 3    # DC 2: Assign next task to VM 3
}
```

**Reward (Local, per DC):**
- Wait time (minimize for this DC's tasks)
- Utilization (maximize for this DC's resources)
- Local queue size (keep short)
- Invalid action penalty

### 13.4 Key Java Components

#### HierarchicalMultiDCGateway

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/HierarchicalMultiDCGateway.java`

**Purpose:** Entry point for Python's hierarchical MARL environment

**Key Methods:**

```java
public class HierarchicalMultiDCGateway {
    private MultiDatacenterSimulationCore simulationCore;
    private SimulationSettings settings;
    private List<DatacenterConfig> datacenterConfigs;

    /**
     * Configure simulation with datacenter definitions from Python
     */
    public void configureSimulation(Map<String, Object> params) {
        this.settings = new SimulationSettings(params);
        this.datacenterConfigs = parseDatacenterConfigs(params);

        // Support both single-DC and multi-DC modes
        boolean multiDcEnabled = getBooleanParam(params, "multi_datacenter_enabled", false);

        if (multiDcEnabled) {
            // Parse list of datacenter configurations
            List<Map<String, Object>> dcList = (List) params.get("datacenters");
            for (Map<String, Object> dcParams : dcList) {
                DatacenterConfig config = parseDatacenterConfig(dcParams);
                datacenterConfigs.add(config);
            }
        } else {
            // Single DC mode - create one default datacenter
            DatacenterConfig config = DatacenterConfig.builder()
                .datacenterId(0)
                .hostsCount(getIntParam(params, "hosts_count", 16))
                // ... other params from config
                .build();
            datacenterConfigs.add(config);
        }
    }

    /**
     * Reset simulation - called at episode start
     */
    public HierarchicalStepResult reset(int seed) {
        simulationCore = new MultiDatacenterSimulationCore(settings, datacenterConfigs);
        simulationCore.resetSimulation();

        // Get initial observations
        ObservationState globalObs = simulationCore.getGlobalObservation();
        Map<Integer, ObservationState> localObs = simulationCore.getLocalObservations();

        return new HierarchicalStepResult(
            globalObs, localObs,
            0.0, new HashMap<>(),  // Initial rewards = 0
            false, false, new HashMap<>()
        );
    }

    /**
     * Execute one hierarchical step
     */
    public HierarchicalStepResult step(
            List<Integer> globalActions,           // DC indices for arriving tasks
            Map<Integer, Integer> localActionsMap  // DC_ID -> VM_ID mappings
    ) {
        return simulationCore.executeHierarchicalStep(globalActions, localActionsMap);
    }

    /**
     * Get count of arriving cloudlets (for action space sizing)
     */
    public int getArrivingCloudletsCount() {
        return simulationCore.getGlobalBroker()
                .getArrivingCloudlets(currentClock, timestepSize)
                .size();
    }
}
```

#### MultiDatacenterSimulationCore

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/MultiDatacenterSimulationCore.java`

**Purpose:** Manages multiple datacenter instances and coordinates hierarchical execution

**Key Methods:**

```java
public class MultiDatacenterSimulationCore {
    private CloudSimPlus simulation;
    private GlobalBroker globalBroker;
    private List<DatacenterInstance> datacenterInstances;
    private List<Cloudlet> allCloudlets;

    /**
     * Reset simulation - create all DCs
     */
    public void resetSimulation() {
        simulation = new CloudSimPlus();

        // Load all cloudlets from workload file
        allCloudlets = loadAllCloudlets();

        // Create datacenter instances
        datacenterInstances = new ArrayList<>();
        for (DatacenterConfig config : datacenterConfigs) {
            DatacenterInstance dc = createDatacenterInstance(config);
            datacenterInstances.add(dc);
        }

        // Create global broker (manages routing)
        globalBroker = new GlobalBroker(simulation, allCloudlets, datacenterInstances);

        simulation.startSync();
    }

    /**
     * Execute one hierarchical step
     */
    public HierarchicalStepResult executeHierarchicalStep(
            List<Integer> globalActions,
            Map<Integer, Integer> localActions
    ) {
        // Phase 1: Global Routing
        List<Cloudlet> arriving = globalBroker.getArrivingCloudlets(currentClock, timestepSize);
        for (int i = 0; i < arriving.size(); i++) {
            Cloudlet cloudlet = arriving.get(i);
            int targetDcIndex = globalActions.get(i);
            globalBroker.routeCloudletToDatacenter(cloudlet, targetDcIndex);
        }

        // Phase 2: Local Scheduling
        for (Map.Entry<Integer, Integer> entry : localActions.entrySet()) {
            int dcId = entry.getKey();
            int vmId = entry.getValue();

            DatacenterInstance dc = datacenterInstances.get(dcId);
            dc.getLocalBroker().assignCloudletToVm(vmId);
        }

        // Phase 3: Advance Simulation
        simulation.runFor(timestepSize);
        currentClock = simulation.clock();

        // Phase 4: Collect Observations
        ObservationState globalObs = getGlobalObservation();
        Map<Integer, ObservationState> localObs = getLocalObservations();

        // Phase 5: Calculate Rewards
        double globalReward = calculateGlobalReward();
        Map<Integer, Double> localRewards = calculateLocalRewards();

        // Phase 6: Check Termination
        boolean terminated = !simulation.isRunning();
        boolean truncated = currentStep >= settings.getMaxEpisodeLength();

        return new HierarchicalStepResult(
            globalObs, localObs,
            globalReward, localRewards,
            terminated, truncated, buildStepInfo()
        );
    }

    /**
     * Get global observation (DC-level aggregated metrics)
     */
    public ObservationState getGlobalObservation() {
        int numDcs = datacenterInstances.size();

        double[] dcGreenPower = new double[numDcs];
        double[] dcQueueSizes = new double[numDcs];
        double[] dcUtilizations = new double[numDcs];
        double[] dcAvailablePes = new double[numDcs];

        for (int i = 0; i < numDcs; i++) {
            DatacenterInstance dc = datacenterInstances.get(i);
            dcGreenPower[i] = dc.getCurrentGreenPowerW(currentClock) / 1000.0;  // kW
            dcQueueSizes[i] = dc.getWaitingCloudletCount();
            dcUtilizations[i] = dc.getAverageHostUtilization();
            dcAvailablePes[i] = dc.getTotalAvailablePes();
        }

        // Get upcoming cloudlets
        List<Cloudlet> upcoming = globalBroker.getArrivingCloudlets(
            currentClock, timestepSize * 2);
        int upcomingCount = upcoming.size();
        int nextCloudletPes = upcoming.isEmpty() ? 0 : (int) upcoming.get(0).getPesNumber();

        return new ObservationState(
            dcUtilizations,  // hostLoads (reused for DC utilizations)
            dcGreenPower,    // vmLoads (reused for green power)
            // ... pack into existing ObservationState structure
        );
    }

    /**
     * Get local observations for all datacenters
     */
    public Map<Integer, ObservationState> getLocalObservations() {
        Map<Integer, ObservationState> observations = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            ObservationState localObs = buildLocalObservation(dc);
            observations.put(dc.getId(), localObs);
        }

        return observations;
    }

    /**
     * Build local observation for a specific datacenter
     */
    private ObservationState buildLocalObservation(DatacenterInstance dc) {
        List<Host> hosts = dc.getHostList();
        List<Vm> vms = dc.getVmPool();
        LoadBalancingBroker localBroker = dc.getLocalBroker();

        // Extract host metrics
        double[] hostLoads = new double[hosts.size()];
        for (int i = 0; i < hosts.size(); i++) {
            hostLoads[i] = hosts.get(i).getCpuPercentUtilization();
        }

        // Extract VM metrics
        double[] vmLoads = new double[vms.size()];
        int[] vmTypes = new int[vms.size()];
        int[] vmAvailablePes = new int[vms.size()];
        for (int i = 0; i < vms.size(); i++) {
            Vm vm = vms.get(i);
            vmLoads[i] = vm.getCpuPercentUtilization();
            vmTypes[i] = determineVmType(vm, dc.getConfig());
            vmAvailablePes[i] = (int) vm.getFreePesNumber();
        }

        // Queue info
        int waitingCloudlets = localBroker.getWaitingCloudletCount();
        int nextCloudletPes = 0;
        if (waitingCloudlets > 0) {
            Cloudlet next = localBroker.peekWaitingCloudlet();
            nextCloudletPes = (int) next.getPesNumber();
        }

        return new ObservationState(
            hostLoads, hostRamUsageRatio,
            vmLoads, vmTypes, vmHostMap,
            infrastructureObs,
            waitingCloudlets, nextCloudletPes, vmAvailablePes,
            vms.size(), hosts.size(),
            0L, 0.0, new int[3], 0
        );
    }

    /**
     * Calculate global reward
     */
    private double calculateGlobalReward() {
        double reward = 0.0;

        // Energy-based rewards
        double totalEnergyWh = 0.0;
        double totalGreenEnergyWh = 0.0;

        for (DatacenterInstance dc : datacenterInstances) {
            double dcPowerW = dc.getHostList().stream()
                .mapToDouble(h -> h.getPowerModel() != null ? h.getPowerModel().getPower() : 0.0)
                .sum();
            double energyWh = dcPowerW * (timestepSize / 3600.0);
            totalEnergyWh += energyWh;

            double greenPowerW = dc.getCurrentGreenPowerW(currentClock);
            double greenEnergyWh = greenPowerW * (timestepSize / 3600.0);
            totalGreenEnergyWh += Math.min(greenEnergyWh, energyWh);
        }

        // Reward for green energy ratio
        double greenRatio = totalEnergyWh > 0 ? totalGreenEnergyWh / totalEnergyWh : 0.0;
        double greenEnergyReward = greenRatio * 10.0;

        // Penalty for total energy
        double energyPenalty = -totalEnergyWh / 1000.0;

        // Load balance penalty
        double loadBalancePenalty = calculateLoadBalancePenalty();

        // Queue size penalty
        int totalWaiting = datacenterInstances.stream()
            .mapToInt(DatacenterInstance::getWaitingCloudletCount)
            .sum();
        double queuePenalty = -totalWaiting * 0.1;

        reward = greenEnergyReward + energyPenalty + loadBalancePenalty + queuePenalty;
        return reward;
    }

    /**
     * Calculate local rewards for each datacenter
     */
    private Map<Integer, Double> calculateLocalRewards() {
        Map<Integer, Double> rewards = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            int dcId = dc.getId();

            // Local objectives
            int waitingCount = dc.getWaitingCloudletCount();
            double queuePenalty = -waitingCount * 0.5;

            double avgUtil = dc.getAverageHostUtilization();
            double utilizationReward = (avgUtil >= 0.5 && avgUtil <= 0.9)
                ? avgUtil * 5.0
                : (avgUtil > 0.9 ? (1.0 - avgUtil) * 2.0 : 0.0);

            double localReward = queuePenalty + utilizationReward;
            rewards.put(dcId, localReward);
        }

        return rewards;
    }
}
```

#### GlobalBroker

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/GlobalBroker.java`

**Purpose:** Routes arriving cloudlets to datacenters (no queuing, immediate routing)

```java
public class GlobalBroker extends DatacenterBrokerSimple {
    private final List<Cloudlet> allCloudlets;  // Sorted by arrival time
    private final List<DatacenterInstance> datacenterInstances;
    private int nextCloudletIndex = 0;

    /**
     * Get cloudlets arriving in time window [currentTime, currentTime + timestep)
     */
    public List<Cloudlet> getArrivingCloudlets(double currentTime, double timestep) {
        List<Cloudlet> arriving = new ArrayList<>();
        double windowEnd = currentTime + timestep;

        while (nextCloudletIndex < allCloudlets.size()) {
            Cloudlet cloudlet = allCloudlets.get(nextCloudletIndex);
            double arrivalTime = cloudlet.getSubmissionDelay();

            if (arrivalTime >= currentTime && arrivalTime < windowEnd) {
                arriving.add(cloudlet);
                nextCloudletIndex++;
            } else if (arrivalTime >= windowEnd) {
                break;  // Future cloudlet
            } else {
                nextCloudletIndex++;  // Past cloudlet (shouldn't happen)
            }
        }

        return arriving;
    }

    /**
     * Route a cloudlet to a specific datacenter
     */
    public boolean routeCloudletToDatacenter(Cloudlet cloudlet, int datacenterIndex) {
        if (datacenterIndex < 0 || datacenterIndex >= datacenterInstances.size()) {
            return false;
        }

        DatacenterInstance targetDC = datacenterInstances.get(datacenterIndex);
        LoadBalancingBroker localBroker = targetDC.getLocalBroker();

        // Add cloudlet to local broker's queue
        boolean success = localBroker.receiveCloudlet(cloudlet);

        if (success) {
            targetDC.incrementCloudletsReceived();
        }

        return success;
    }
}
```

#### DatacenterInstance

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/DatacenterInstance.java`

**Purpose:** Encapsulates all components for a single datacenter

```java
public class DatacenterInstance {
    private final DatacenterConfig config;

    // CloudSim entities
    private Datacenter datacenter;
    private LoadBalancingBroker localBroker;

    // Infrastructure
    private List<Host> hostList;
    private List<Vm> vmPool;

    // Green energy
    private GreenEnergyProvider greenEnergyProvider;

    // Statistics
    private int cloudletsReceived = 0;
    private int cloudletsCompleted = 0;

    public int getId() {
        return config.getDatacenterId();
    }

    public double getCurrentGreenPowerW(double simulationTime) {
        if (greenEnergyProvider == null) return 0.0;
        return greenEnergyProvider.getCurrentPowerW(simulationTime);
    }

    public int getWaitingCloudletCount() {
        return localBroker != null ? localBroker.getWaitingCloudletCount() : 0;
    }

    public double getAverageHostUtilization() {
        return hostList.stream()
            .mapToDouble(Host::getCpuPercentUtilization)
            .average()
            .orElse(0.0);
    }

    public int getTotalAvailablePes() {
        return vmPool.stream()
            .mapToInt(vm -> (int) vm.getFreePesNumber())
            .sum();
    }
}
```

### 13.5 Python Environment

**Location:** `drl-manager/hierarchical_multidc_env.py`

```python
class HierarchicalMultiDCEnv(gym.Env):
    """
    Two-level hierarchical MARL environment
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_datacenters = len(config.get("datacenters", [{"datacenter_id": 0}]))
        self.max_arriving_cloudlets = config.get("max_arriving_cloudlets", 50)

        # Connect to Java
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))
        self.java_env = self.gateway.entry_point

        # Configure simulation
        self.java_env.configureSimulation(config)

        # Define observation spaces
        self.global_observation_space = spaces.Dict({
            "dc_green_power": spaces.Box(0.0, 10000.0, (self.num_datacenters,), np.float32),
            "dc_queue_sizes": spaces.Box(0, 10000, (self.num_datacenters,), np.float32),
            "dc_utilizations": spaces.Box(0.0, 1.0, (self.num_datacenters,), np.float32),
            "dc_available_pes": spaces.Box(0, 1000, (self.num_datacenters,), np.float32),
            "upcoming_cloudlets_count": spaces.Discrete(self.max_arriving_cloudlets + 1),
            "next_cloudlet_pes": spaces.Discrete(100),
        })

        self.local_observation_space = spaces.Dict({
            "host_loads": spaces.Box(0.0, 1.0, (max_hosts,), np.float32),
            "vm_loads": spaces.Box(0.0, 1.0, (max_vms,), np.float32),
            "vm_types": spaces.Box(0, 3, (max_vms,), np.int32),
            "vm_available_pes": spaces.Box(0, 100, (max_vms,), np.int32),
            "waiting_cloudlets": spaces.Discrete(10000),
            "next_cloudlet_pes": spaces.Discrete(100),
        })

        # Action spaces
        self.global_action_space = spaces.MultiDiscrete([self.num_datacenters] * self.max_arriving_cloudlets)
        self.local_action_space = spaces.Discrete(max_vms)

    def reset(self, seed=None, options=None):
        result = self.java_env.reset(seed if seed is not None else 0)
        observations = self._parse_hierarchical_observation(result)
        info = self._parse_info(result)
        return observations, info

    def step(self, action: Dict[str, Any]):
        """
        action = {
            'global': [0, 1, 0, 2, 1],  # DC indices for arriving cloudlets
            'local': {0: 5, 1: 12, 2: 3}  # DC_ID -> VM_ID mappings
        }
        """
        global_actions = action.get("global", [])
        local_actions_map = action.get("local", {})

        # Get actual arriving count
        num_arriving = self.java_env.getArrivingCloudletsCount()
        if len(global_actions) > num_arriving:
            global_actions = global_actions[:num_arriving]

        # Execute step
        result = self.java_env.step(global_actions, local_actions_map)

        # Parse results
        observations = self._parse_hierarchical_observation(result)
        rewards = self._parse_hierarchical_rewards(result)
        terminated = result.isTerminated()
        truncated = result.isTruncated()
        info = self._parse_info(result)

        return observations, rewards, terminated, truncated, info

    def _parse_hierarchical_observation(self, result):
        # Parse global observation
        global_obs_java = result.getGlobalObservation()
        global_obs = {
            "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),
            "dc_queue_sizes": np.array(global_obs_java.getHostRamUsageRatio(), dtype=np.float32),
            "dc_utilizations": np.array(global_obs_java.getHostLoads(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getVmAvailablePes(), dtype=np.float32),
            "upcoming_cloudlets_count": global_obs_java.getWaitingCloudlets(),
            "next_cloudlet_pes": global_obs_java.getNextCloudletPes(),
        }

        # Parse local observations
        local_obs_map_java = result.getLocalObservations()
        local_obs = {}
        for dc_id in range(self.num_datacenters):
            if dc_id in local_obs_map_java:
                local_obs_java = local_obs_map_java[dc_id]
                local_obs[dc_id] = {
                    "host_loads": np.array(local_obs_java.getHostLoads(), dtype=np.float32),
                    "vm_loads": np.array(local_obs_java.getVmLoads(), dtype=np.float32),
                    "vm_types": np.array(local_obs_java.getVmTypes(), dtype=np.int32),
                    "vm_available_pes": np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32),
                    "waiting_cloudlets": local_obs_java.getWaitingCloudlets(),
                    "next_cloudlet_pes": local_obs_java.getNextCloudletPes(),
                }

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_hierarchical_rewards(self, result):
        global_reward = result.getGlobalReward()
        local_rewards_java = result.getLocalRewards()
        local_rewards = {dc_id: local_rewards_java.get(dc_id, 0.0)
                        for dc_id in range(self.num_datacenters)}

        return {
            "global": global_reward,
            "local": local_rewards
        }
```

### 13.6 Configuration Example

**Multi-Datacenter Configuration in `config.yml`:**

```yaml
experiment_multidc:
  # Enable multi-datacenter mode
  multi_datacenter_enabled: true

  # Global settings
  simulation_name: "MultiDC_GreenRouting"
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/poisson_04_3600.csv"
  max_episode_length: 3600
  max_arriving_cloudlets: 50

  # RL Algorithm
  algorithm: "HierarchicalPPO"
  timesteps: 1000000
  learning_rate: 0.0003

  # Datacenter configurations
  datacenters:
    - datacenter_id: 0
      name: "DC_California"
      hosts_count: 16
      host_pes: 16
      initial_s_vm_count: 15
      initial_m_vm_count: 8
      initial_l_vm_count: 4
      green_energy_enabled: true
      turbine_id: 57  # California wind farm
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

    - datacenter_id: 1
      name: "DC_Texas"
      hosts_count: 20
      host_pes: 16
      initial_s_vm_count: 20
      initial_m_vm_count: 10
      initial_l_vm_count: 5
      green_energy_enabled: true
      turbine_id: 42  # Texas wind farm
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

    - datacenter_id: 2
      name: "DC_Ireland"
      hosts_count: 12
      host_pes: 16
      initial_s_vm_count: 10
      initial_m_vm_count: 6
      initial_l_vm_count: 3
      green_energy_enabled: true
      turbine_id: 88  # Ireland wind farm
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

  # Reward weights
  global_green_energy_coef: 5.0
  global_energy_penalty_coef: 1.0
  global_load_balance_coef: 2.0
  global_queue_penalty_coef: 0.5

  local_wait_time_coef: 1.0
  local_utilization_coef: 0.5
  local_queue_penalty_coef: 0.8
```

### 13.7 Training Hierarchical Agents

**Approach 1: Shared Policy (Parameter Sharing)**

All local agents share the same policy network:

```python
from sb3_contrib import RecurrentPPO

env = HierarchicalMultiDCEnv(config)

# Single global agent for routing
global_model = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=env.global_env,  # Wrapper that only exposes global obs/action
    learning_rate=0.0003
)

# Shared local agent for all DCs
local_model = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=env.local_env,  # Wrapper that rotates through DCs
    learning_rate=0.0003
)

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        # Global agent decides routing
        global_action = global_model.predict(obs["global"])

        # Local agents decide scheduling (shared policy)
        local_actions = {}
        for dc_id in range(env.num_datacenters):
            local_action = local_model.predict(obs["local"][dc_id])
            local_actions[dc_id] = local_action

        # Execute step
        action = {"global": global_action, "local": local_actions}
        obs, rewards, done, info = env.step(action)

        # Update policies
        global_model.learn(obs["global"], rewards["global"])
        for dc_id in range(env.num_datacenters):
            local_model.learn(obs["local"][dc_id], rewards["local"][dc_id])
```

**Approach 2: Independent Policies**

Each datacenter has its own local agent:

```python
# Global agent
global_agent = RecurrentPPO(...)

# Independent local agents
local_agents = {
    0: RecurrentPPO(...),
    1: RecurrentPPO(...),
    2: RecurrentPPO(...)
}

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        global_action = global_agent.predict(obs["global"])

        local_actions = {}
        for dc_id, agent in local_agents.items():
            local_actions[dc_id] = agent.predict(obs["local"][dc_id])

        obs, rewards, done, info = env.step({
            "global": global_action,
            "local": local_actions
        })

        # Update all agents independently
        global_agent.learn(...)
        for dc_id, agent in local_agents.items():
            agent.learn(obs["local"][dc_id], rewards["local"][dc_id])
```

### 13.8 Running Multi-Datacenter Experiments

**Step 1: Start Java Gateway**
```bash
cd cloudsimplus-gateway

# Use HierarchicalMultiDCGateway instead of Main
./gradlew run --args="hierarchical"
# OR edit Main.java to start HierarchicalMultiDCGateway

# Wait for: "Hierarchical MultiDC Gateway Server started on port 25333"
```

**Step 2: Start Training**
```bash
cd drl-manager

python train_hierarchical.py --experiment experiment_multidc
```

### 13.9 Key Differences from Single-DC Mode

| Aspect | Single-DC Mode | Multi-DC Mode |
|--------|----------------|---------------|
| **Entry Point** | `LoadBalancerGateway` | `HierarchicalMultiDCGateway` |
| **Decision Levels** | 1 (VM selection) | 2 (DC routing + VM selection) |
| **Agents** | 1 agent | 1 global + N local agents |
| **Observation** | Single dict | Hierarchical: global + local per DC |
| **Action** | Integer (VM ID) | Dict: global list + local dict |
| **Reward** | Single scalar | Dict: global + local per DC |
| **Broker** | LoadBalancingBroker | GlobalBroker + LocalBrokers |
| **Green Energy** | Optional per DC | Optimized globally |
| **Scalability** | Limited to ~100 VMs | Scales to 1000s VMs across DCs |

### 13.10 Use Cases for Multi-Datacenter Mode

1. **Green Energy Optimisation**: Route tasks to datacenters with high renewable energy availability
2. **Geographic Load Balancing**: Distribute workload across regions to reduce latency
3. **Cost Optimisation**: Route tasks to datacenters with lower electricity costs
4. **Fault Tolerance**: Redirect tasks from overloaded/failing datacenters

---

## 14. Extending the System

### 14.1 Adding New Observation Features

**Example: Add "average task size in queue"**

**Step 1: Java - Compute metric in LoadBalancerGateway**

```java
// In getCurrentState()
double avgQueueTaskSize = calculateAverageQueueMi();  // New helper method

// Add to ObservationState constructor
return new ObservationState(
    vmLoads, vmTypes, vmAvailablePes,
    waitingCloudlets, nextCloudletPes, nextCloudletMi,
    nextCloudletWaitTime, queuePesDistribution,
    completedLast10,
    avgQueueTaskSize,  // ← New parameter
    activeVms.size()
);
```

**Step 2: Java - Update ObservationState class**
```java
public class ObservationState {
    // Add field
    private final double avgQueueTaskSize;

    // Update constructor
    public ObservationState(..., double avgQueueTaskSize, int actualVmCount) {
        // ...
        this.avgQueueTaskSize = avgQueueTaskSize;
    }

    // Add getter
    public double getAvgQueueTaskSize() {
        return avgQueueTaskSize;
    }
}
```

**Step 3: Python - Update observation space**

```python
# In LoadBalancingEnv.__init__()
self.observation_space = spaces.Dict({
    # ... existing keys ...
    "avg_queue_task_size": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # New key
})
```

**Step 4: Python - Extract in _get_obs()**
```python
def _get_obs(self, java_obs_state):
    # ... existing extraction ...

    # Extract new field
    avg_queue_task_size = np.array([java_obs_state.getAvgQueueTaskSize()], dtype=np.float32)

    obs_dict = {
        # ... existing keys ...
        "avg_queue_task_size": avg_queue_task_size  # Add to dict
    }
    return obs_dict
```

**Step 5: Rebuild and restart**
```bash
cd cloudsimplus-gateway
./gradlew build  # Rebuild Java
./gradlew run    # Restart gateway

cd ../drl-manager
python mnt/train.py --experiment my_experiment  # Retrain
```

---

### 14.2 Adding New Reward Components

**Example: Add "energy efficiency reward"**

**Step 1: Java - Calculate energy metric**
```java
// In calculateReward()
double currentPowerWatts = calculateTotalPowerConsumption();
double energyEfficiency = completedTasksLastStep / (currentPowerWatts + 1.0);
double rewardEnergy = energyEfficiency * config.rewardEnergyCoef;

// Add to total
double totalReward = rewardWaitTime + rewardUnutilization
                   + rewardQueue + rewardInvalidAction
                   + rewardEnergy;  // ← Add new component

// Log component
lastRewardComponents.put("reward_energy", rewardEnergy);
```

**Step 2: Config - Add weight parameter**
```yaml
# In config.yml
common:
  reward_energy_coef: 0.1  # New weight
```

**Step 3: Python - Log new metric**
```python
# In train.py
info_keywords_to_log = (
    "reward_wait_time",
    "reward_unutilization",
    "reward_queue_penalty",
    "reward_invalid_action",
    "reward_energy",  # Add new component
    # ...
)
```

---

### 14.3 Implementing Custom Action Masking

**Example: Mask actions for VMs that can't fit next task**

**Step 1: Java - Add action mask to ObservationState**
```java
public class ObservationState {
    private final boolean[] actionMask;  // New field

    public boolean[] getActionMask() {
        return actionMask;
    }
}

// In getCurrentState()
boolean[] mask = new boolean[maxVms + 1];
mask[0] = true;  // No-op always valid
for (int i = 0; i < activeVms.size(); i++) {
    Vm vm = activeVms.get(i);
    mask[i+1] = vm.getFreePesNumber() >= nextCloudlet.getNumberOfPes();
}
```

**Step 2: Python - Use MaskablePPO**
```python
from sb3_contrib import MaskablePPO

# In LoadBalancingEnv._get_obs()
action_mask = np.array(java_obs_state.getActionMask(), dtype=np.bool_)
obs_dict["action_mask"] = action_mask

# In train.py
model = MaskablePPO(
    policy="MultiInputPolicy",
    env=env,
    # ... other params
)
```

Now the agent can only select valid actions!

---



## Results

![{86A69D4E-522F-4D67-B004-395FD0FB3D0B}](C:\Users\admin\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{86A69D4E-522F-4D67-B004-395FD0FB3D0B}.png)

![{2A950115-3164-42FA-A13E-3BDD5316616D}](C:\Users\admin\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{2A950115-3164-42FA-A13E-3BDD5316616D}.png)

![{3B3D99C8-08B8-439A-8E47-0E9CDDF55543}](C:\Users\admin\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{3B3D99C8-08B8-439A-8E47-0E9CDDF55543}.png)

## Conclusion

This document provides a comprehensive overview of the RL-CloudSim load balancing system. The key takeaway is understanding how **Py4J bridges Python's RL ecosystem with Java's cloud simulation**, enabling sophisticated reinforcement learning for cloud resource management.

For questions or contributions, please refer to the project README or contact the development team.
