### AGENTS.md (for Cursor/AI contributors)

This repo is a **CloudSim Plus + Py4J** simulation backend with a **Python RL manager** (Gymnasium/SB3 for single-DC, PettingZoo/RLlib for multi-DC) focused on **green energy / carbon-aware scheduling**.

### Golden rules
- **Do not edit** `drl-manager/Code/` (legacy/externals). Prefer changes elsewhere unless explicitly requested.
- **Prefer small, verifiable changes**: locate call sites, patch, then run the nearest compile/quick check.
- **When changing Java gateway code**, always verify with:

```bash
cd cloudsimplus-gateway
./gradlew -q compileJava
```

### Key entrypoints (what to run)
#### Multi-DC (recommended, hierarchical MARL)
- **Java gateway**: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/MainMultiDC.java`
- **Python entry**: `drl-manager/entrypoint_pettingzoo.py`

#### Single-DC (SB3)
- **Java gateway**: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/Main.java`
- **Python entry**: `drl-manager/entrypoint.py`

### Configuration source of truth
- Experiment config lives in **root** `config.yml`.
- Python loads config and decides which training/eval path to execute.

### Data flow (high-level)
#### Multi-DC step loop (Java side)
- `MultiDatacenterSimulationCore.executeGlobalRouting(...)` calls:
  - `GlobalBroker.processArrivingCloudlets(currentClock, timestepSize)` to enqueue arrivals
  - `GlobalBroker.getBatchForRouting(batchSize)` to get a fixed-size batch
  - `GlobalBroker.routeCloudletToDatacenter(cloudlet, targetDcIndex)` per cloudlet

#### Arrival time semantics (important)
- In this project, **CSV `arrival_time` is mapped to CloudSim `Cloudlet.submissionDelay`**.
  - Reader: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/utils/WorkloadFileReader.java`
  - DTO→Cloudlet: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/common/CloudletDescriptor.java`
- `GlobalBroker` treats `cloudlet.getSubmissionDelay()` as the arrival timestamp when deciding “arriving cloudlets”.

### Energy / carbon metrics (where to look)
- Datacenter cumulative snapshot DTO: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/DatacenterEnergyMetrics.java`
- Per-timestep delta DTO (for reward): `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/EnergyMetricsDelta.java`
- Core update logic: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/DatacenterInstance.java`
  - Defensive behavior exists to avoid stale `latestEnergyDelta` when no hosts/time doesn’t advance.

### Conventions & common pitfalls
- **Indices vs IDs**: Many arrays use **datacenterIndex** (0..N-1). Do not assume it equals an arbitrary datacenterId unless you verified mapping.
- **Logging**: SLF4J `{}` placeholders do **not** support `:.2f` formatting. Use `String.format` if you need fixed decimals.
- **Immutability**: `GlobalObservationState` returns **copies** of internal arrays. Keep this pattern (avoid leaking mutable arrays).

### Quick “where is X implemented?”
- **Global routing queue**: `GlobalBroker`
- **Local broker / cloudlet submission**: `giu.edu.cspg.singledc.LoadBalancingBroker`
- **Observation objects**:
  - Global: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/GlobalObservationState.java`
  - Local env side (Python): `drl-manager/gym_cloudsimplus/envs/`

### Output locations (typical)
- Python logs/checkpoints are under `drl-manager/logs/` (exact subfolders depend on SB3 vs RLlib).
- Java gateway logs go to stdout/stderr (captured by your run environment).


