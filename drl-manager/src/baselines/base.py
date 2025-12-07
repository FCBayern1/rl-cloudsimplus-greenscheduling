from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class GlobalScheduler(ABC):
    """Global Scheduler Base Class： Select Target DC for each Cloudlet in the Batch"""

    def __init__(self, num_datacenters: int, batch_size: int):
        self.num_datacenters = num_datacenters
        self.batch_size = batch_size

    @abstractmethod
    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        """
        Select Datacenter for Cloudlets

        Args:
            global_obs:
        Returns:
            List[int]: DC Index for each cloudlet，length = batch_size
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset(self):
        pass


class LocalScheduler(ABC):
    """Local Scheduler Base Class: Select VM for each cloudlet within Datacenter"""

    def __init__(self, num_vms: int):
        self.num_vms = num_vms

    @abstractmethod
    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        Select VM for each cloudlet within Datacenter

        Args:
            local_obs: DC Local Observation
                - vm_loads: VM Load for each cloudlet
                - vm_available_pes: Available PEs for each VM
                - waiting_cloudlets: Waiting Cloudlets Queue Length
                - next_cloudlet_pes: PE Demand for next cloudlet which is waited being scheduled
            action_mask: shape=(num_vms+1,)
                - mask[0]=True: No Assign
                - mask[i]=True: Assign to VM i-1

        Returns:
            int: VM action (0=NoAssign, 1~N=VM index)
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset(self):
        pass
