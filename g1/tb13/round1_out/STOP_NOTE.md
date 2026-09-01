# Round 1 STOP and the zero-carbon diagnostic that followed

Round 1 stopped in Phase A because no single causal blind honoured the contract on every
required instance. The exact oracle was never solved on these instances and no EVPI was
computed. Phase B is not resumed.

A post-STOP diagnostic then ran a pure feasibility model and a per-arm failure matrix,
reading no carbon. Its finding:

    one unique workload is itself infeasible, appearing as nine grid cells across nine
    green configurations; two further unique workloads are feasible offline yet defeat all
    four online blinds. Every forced dispatch was triggered by the delay budget, not by a
    deadline, and collided with capacity at that moment.

The 1,296 grid cells carry only 272 unique workloads, so counting cells overstates the
number of independent loads. Both faults are addressed together in Round1-v2, which is a
separate registration: the generator did not guarantee a feasible region, and the blind
family lacked a contract-safe reservation policy.
