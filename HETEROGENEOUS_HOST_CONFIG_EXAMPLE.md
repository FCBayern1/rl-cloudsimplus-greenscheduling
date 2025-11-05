# Heterogeneous Host Configuration Guide

## Overview

The system now supports flexible heterogeneous host composition using a simple `Map<String, Integer>` approach.

## Configuration Format

### YAML Configuration

```yaml
multi_datacenter:
  enabled: true

  datacenters:
    # Example 1: Mixed deployment
    - datacenter_id: 0
      datacenter_name: "US-East-Mixed"

      # Define host composition as simple key-value pairs
      host_profiles:
        MEDIUM: 10                    # 10 x Dell R720 (16 cores, 345W)
        HIGH_PERFORMANCE: 5           # 5 x Cisco UCS C240 (24 cores, 476W)
        LOW_POWER: 3                  # 3 x HP DL360 G7 (8 cores, 208W)

      # VM configuration
      small_vm_pes: 2
      small_vm_ram: 8192
      small_vm_bw: 1000
      small_vm_storage: 4000
      medium_vm_multiplier: 2
      large_vm_multiplier: 4

      initial_small_vm_count: 10
      initial_medium_vm_count: 5
      initial_large_vm_count: 3

      # Green energy
      green_energy_enabled: true
      turbine_id: 57
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

    # Example 2: Modern AMD EPYC servers
    - datacenter_id: 1
      datacenter_name: "EU-West-ModernAMD"

      host_profiles:
        SPEC_ASUS_RS500A: 20          # 64 cores, 214W (super efficient!)
        SPEC_ASUS_RS700A: 5           # 128 cores, 430W

      # ... VM and green energy config ...

    # Example 3: Legacy datacenter
    - datacenter_id: 2
      datacenter_name: "Asia-Pacific-Legacy"

      host_profiles:
        SPEC_ACER_R520: 30            # 8 cores, 269W (old, inefficient)
        SPEC_ACER_AR360: 10           # 16 cores, 315W

      # ... VM and green energy config ...
```

## Available Host Profiles

### Generic Profiles
- **LOW_POWER**: HP ProLiant DL360 G7 (8 cores, 208W peak, 58W idle)
- **MEDIUM**: Dell PowerEdge R720 (16 cores, 345W peak, 98W idle)
- **HIGH_PERFORMANCE**: Cisco UCS C240 M4 (24 cores, 476W peak, 142W idle)
- **ULTRA_HIGH**: HPE ProLiant DL380 Gen10 (32 cores, 634W peak, 194W idle)

### SPEC Real Server Profiles
- **SPEC_ACER_R520**: Acer Altos R520 (8 cores, 269W peak, 155W idle) - Legacy
- **SPEC_ACER_AR360**: Acer AR360 F2 (16 cores, 315W peak, 69.4W idle)
- **SPEC_ASUS_RS720_E9**: ASUSTeK RS720-E9/RS8 (56 cores, 385W peak, 48.2W idle) - Modern
- **SPEC_ASUS_RS500A**: ASUSTeK RS500A-E10-PS4 (64 cores, 214W peak, 51.4W idle) - AMD EPYC
- **SPEC_ASUS_RS700A**: ASUSTeK RS700A-E9-RS4V2 (128 cores, 430W peak, 106W idle) - Dual EPYC

## Java Code Usage

### Creating a Datacenter Config

```java
// Using builder with @Singular for clean API
DatacenterConfig config = DatacenterConfig.builder()
    .datacenterId(0)
    .datacenterName("MyDatacenter")
    // Add host profiles one by one
    .hostProfile("MEDIUM", 10)
    .hostProfile("HIGH_PERFORMANCE", 5)
    .hostProfile("SPEC_ASUS_RS500A", 3)
    // VM config
    .smallVmPes(2)
    .smallVmRam(8192)
    .smallVmBw(1000)
    .smallVmStorage(4000)
    .mediumVmMultiplier(2)
    .largeVmMultiplier(4)
    .initialSmallVmCount(10)
    .initialMediumVmCount(5)
    .initialLargeVmCount(3)
    // Green energy
    .greenEnergyEnabled(true)
    .turbineId(57)
    .windDataFile("windProduction/sdwpf_2001_2112_full.csv")
    .vmStartupDelay(0.0)
    .vmShutdownDelay(0.0)
    .build();

// Validate configuration
config.validate();

// Create hosts
List<Host> hosts = DatacenterSetup.createHostsForDatacenter(config);
```

### Using the Default Heterogeneous Config

```java
// Quick setup with predefined mixed hosts
DatacenterConfig config = DatacenterConfig.createDefaultHeterogeneous(0);
// Creates: 10 x MEDIUM, 5 x HIGH_PERFORMANCE, 3 x LOW_POWER
```

## Python Configuration Parsing

```python
def parse_datacenter_config(dc_config_dict):
    """Parse datacenter configuration from YAML"""

    builder = gateway.jvm.giu.edu.cspg.common.DatacenterConfig.builder()

    builder.datacenterId(dc_config_dict['datacenter_id'])
    builder.datacenterName(dc_config_dict['datacenter_name'])

    # Parse host profiles (dict: {"MEDIUM": 10, "HIGH_PERFORMANCE": 5})
    if 'host_profiles' in dc_config_dict:
        for profile_name, count in dc_config_dict['host_profiles'].items():
            builder.hostProfile(profile_name, count)

    # VM configuration
    builder.smallVmPes(dc_config_dict.get('small_vm_pes', 2))
    builder.smallVmRam(dc_config_dict.get('small_vm_ram', 8192))
    # ... other fields ...

    return builder.build()
```

## Migration from Legacy Format

### Old Format (Deprecated)
```yaml
datacenter_id: 0
hosts_count: 16
host_pes: 16
host_pe_mips: 50000
host_ram: 65536
host_bw: 50000
host_storage: 100000
```

### New Format (Recommended)
```yaml
datacenter_id: 0
host_profiles:
  MEDIUM: 16  # Much cleaner!
```

## Benefits

1. **Simplicity**: No extra classes needed, just `Map<String, Integer>`
2. **Flexibility**: Mix any number of different host types
3. **Realism**: Use real server profiles from SPEC benchmarks
4. **Energy Accuracy**: Each profile includes real power consumption data
5. **Easy Configuration**: Clean YAML format
6. **Type Safety**: Validates profile names at runtime

## Research Use Cases

### Energy Efficiency Study
Compare old vs new servers:
```yaml
host_profiles:
  SPEC_ACER_R520: 20      # Old: 57.6% idle power ratio
  SPEC_ASUS_RS500A: 20    # New: 24.0% idle power ratio
```

### Cost Optimization
Mix cheap low-power servers with expensive high-performance:
```yaml
host_profiles:
  LOW_POWER: 30           # Handle baseline load
  HIGH_PERFORMANCE: 5     # Handle peak load
```

### Real Datacenter Simulation
Model actual infrastructure:
```yaml
host_profiles:
  SPEC_ASUS_RS720_E9: 50  # Core production servers
  MEDIUM: 100             # Edge nodes
  SPEC_ACER_AR360: 20     # Dev/test environment
```

## Validation

The system automatically validates:
- Profile names exist
- Host counts are non-negative
- At least one host is configured
- Total host count > 0

Invalid configurations will throw `IllegalArgumentException` with helpful error messages.

## Summary

This simple Map-based approach provides maximum flexibility with minimal code complexity. No need for extra classes like `HostComposition` - just a clean key-value mapping of profile names to host counts!
