import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

# Mock logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_action_mask():
    print("Verifying Action Mask Logic...")

    # 1. Instantiate Env (Mock config)
    config = {
        "multi_datacenter_enabled": True,
        "datacenters": [
            {"datacenter_id": 0, "hosts_count": 2, "initial_s_vm_count": 2, "initial_m_vm_count": 0, "initial_l_vm_count": 0}
        ],
        "py4j_port": 25333 # Won't connect, but needed for init
    }
    
    # Mocking the environment to avoid Java connection
    class MockEnv(HierarchicalMultiDCEnv):
        def __init__(self, config):
            # Skip super init to avoid Java connection
            self.config = config
            self.num_datacenters = 1
            self.local_action_space = type('obj', (object,), {'n': 3}) # 2 VMs + 1 NoAssign
            self.dc_vm_counts = [2]
            self.max_vms = 2
            self.java_env = "Mock" # Bypass None check
            
        def _get_dc_vm_count(self, dc_id):
            return 2

    env = MockEnv(config)

    # 2. Set Mock Observation (Case 3: Queue > 0, No VM has enough PEs)
    # VM 0: 2 PEs, VM 1: 2 PEs
    # Next Cloudlet: 4 PEs (Impossible)
    env.last_observations = {
        "local": {
            0: {
                "vm_available_pes": np.array([2, 2]),
                "waiting_cloudlets": 5,
                "next_cloudlet_pes": 4
            }
        }
    }

    # 3. Get Mask
    mask = env.get_local_action_masks(0)
    print(f"Mask: {mask}")

    # 4. Verify Logic
    # Expected: mask[0] (NoAssign) = False, mask[1:] (VMs) = True
    
    if mask[0] == False:
        print("[PASS] NoAssign is DISALLOWED")
    else:
        print("[FAIL] NoAssign is ALLOWED (Should be Disallowed)")

    if np.all(mask[1:] == True):
        print("[PASS] All VMs are ALLOWED")
    else:
        print(f"[FAIL] Not all VMs are allowed: {mask[1:]}")

    if mask[0] == False and np.all(mask[1:] == True):
        print("\nSUCCESS: Action mask logic matches loadbalancing_env.py!")
        return True
    else:
        print("\nFAILURE: Action mask logic mismatch.")
        return False

if __name__ == "__main__":
    verify_action_mask()
