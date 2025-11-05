package giu.edu.cspg.common;

import lombok.Getter;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyRoundRobin;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A custom VM allocation policy that allows placing a VM onto a specific Host,
 * identified by an ID embedded in the VM's description during the creation
 * request.
 * If no specific Host ID is found in the description, it falls back to the
 * standard VmAllocationPolicyRoundRobin behavior.
 */
public class VmAllocationPolicyCustom extends VmAllocationPolicyRoundRobin {
    private static final Logger logger = LoggerFactory.getLogger(VmAllocationPolicyCustom.class.getSimpleName());

    public VmAllocationPolicyCustom() {
        super();
    }

    /**
     * Allocates a Host for a given VM.
     * It checks if the VM's description contains a specific Host ID suffix (e.g.,
     * "S-host5").
     * If found, it attempts to allocate the VM to that specific Host.
     * Otherwise, it delegates to the superclass's allocation method (likely
     * first-fit).
     * After successful allocation to a specific host, the description suffix is
     * removed.
     *
     * @param vm The VM to allocate a Host for.
     * @return An Optional containing the allocated Host if successful, otherwise an
     *         empty Optional.
     */
    @Override
    public HostSuitability allocateHostForVm(Vm vm) {
        // Check if the Datacenter has been set (important!)
        if (getDatacenter() == Datacenter.NULL || getDatacenter().getHostList().isEmpty()) {
            logger.error("Datacenter is null or has no hosts when trying to allocate VM {}. Cannot place VM.",
                    vm.getId());
            return HostSuitability.NULL; // Indicate failure: No suitable host found
        }

        // Check if the description contains the special format "TYPE-hostID"
        final String vmDescription = vm.getDescription();
        final int separatorIndex = vmDescription.indexOf('-');

        Host targetHost = Host.NULL;
        String originalVmType = vmDescription; // Default to full description

        if (separatorIndex != -1 && separatorIndex < vmDescription.length() - 1) {
            // Potential host ID suffix found
            String typePart = vmDescription.substring(0, separatorIndex);
            String idPart = vmDescription.substring(separatorIndex + 1);
            try {
                long hostId = Long.parseLong(idPart);
                // Attempt to find the specific host
                targetHost = getDatacenter().getHostById(hostId);
                if (targetHost != Host.NULL) {
                    originalVmType = typePart; // Store the actual type (S, M, L)
                    logger.debug("Attempting placement of VM {} (type {}) onto specific Host {}", vm.getId(),
                            originalVmType, hostId);
                } else {
                    logger.warn(
                            "VM {} description specified Host ID {} but it was not found. Falling back to default allocation.",
                            vm.getId(), hostId);
                    // targetHost remains Host.NULL, fall back to default policy
                }
            } catch (NumberFormatException e) {
                logger.warn("Could not parse Host ID from VM {} description '{}'. Falling back to default allocation.",
                        vm.getId(), vmDescription);
                // targetHost remains Host.NULL, fall back to default policy
            }
        } else {
            logger.trace("VM {} description '{}' doesn't specify target host. Using default allocation.", vm.getId(),
                    vmDescription);
            // targetHost remains Host.NULL, fall back to default policy
        }

        HostSuitability suitability;
        if (targetHost != Host.NULL) {
            // Attempt to allocate to the specific host
            suitability = allocateHostForVm(vm, targetHost);
        } else {
            // Fallback to the superclass (VmAllocationPolicySimple) behavior
            logger.trace("Using VmAllocationPolicySimple fallback for VM {}.", vm.getId());
            suitability = super.allocateHostForVm(vm);
        }

        // If allocation was successful, clean up the description (remove the -hostID
        // part)
        // and set the final description to just the VM type.
        if (suitability.getHost() != Host.NULL) {
            vm.setDescription(originalVmType); // Set description to S, M, or L
            logger.debug("VM {} successfully allocated to Host {}. Final VM description: '{}'", vm.getId(),
                    suitability.getHost().getId(), vm.getDescription());
        } else {
            logger.warn("Failed to allocate Host for VM {} (type {}).", vm.getId(), originalVmType);
            // VM description might still contain the suffix if allocation failed, which is
            // acceptable.
        }

        return suitability;
    }

    /**
     * Tries to allocate the VM to the given Host.
     * This method is called by the overridden allocateHostForVm(Vm) method
     * when a specific target host is identified.
     * 
     * @param vm   the VM to allocate a host to
     * @param host the target host
     * @return An Optional containing the Host if the VM was successfully allocated,
     *         Optional.empty otherwise.
     */
    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        // Delegate the check and actual allocation to the superclass implementation
        // This reuses the resource checking logic (PEs, RAM, BW) from
        // VmAllocationPolicySimple.
        HostSuitability suitability = super.allocateHostForVm(vm, host);

        if (suitability.getHost() != Host.NULL) {
            logger.trace("Host {} is suitable for VM {}. Allocation successful.", host.getId(), vm.getId());
        } else {
            logger.warn(
                    "Target Host {} is NOT suitable for VM {}. Requirements: PEs={}, RAM={}, BW={}. Host Free: PEs={}, RAM={}, BW={}",
                    host.getId(), vm.getId(),
                    vm.getPesNumber(), vm.getRam().getCapacity(), vm.getBw().getCapacity(),
                    host.getFreePesNumber(), host.getRam().getAvailableResource(), host.getBw().getAvailableResource());
        }
        return suitability;
    }
}
