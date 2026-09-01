import sys, time, csv
import numpy as np
sys.path.insert(0,'/home/joshua/rl-cloudsimplus-greenscheduling/g1/tb13')
from exact_oracle import Scenario, solve, nowait_schedule
WD='/home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/simplified'
def ser(t,y=2021):
    return np.array([float(r['power_kw'] or 0) for r in csv.DictReader(open(f'{WD}/Turbine_{t}_{y}.csv'))])
s12,s36,s96 = ser(12),ser(36),ser(96)
rng=np.random.default_rng(0)
print('exact stage timing, 3 DC, real SDWPF slices')
for n_jobs,T in ((8,36),(10,36),(12,36),(12,48),(16,36)):
    off=20000
    green=np.stack([s12[off:off+T]/1.5, s36[off:off+T]/1.5, s96[off:off+T]/1.5])+120.0
    a=rng.integers(0,max(1,T//4),n_jobs); r=rng.integers(2,5,n_jobs)
    sc=Scenario(green_w=green, static_w=[120.]*3, brown_factor=[0.3,0.5,0.7],
                green_factor=[0.02]*3, cap_pes=[8,8,8], arrival=a, runtime=r,
                pes=[2]*n_jobs, deadline=[T]*n_jobs, dyn_w_per_pe=2.5406,
                per_job_wait_max=min(24,T-5), budget_total=6*n_jobs)
    t0=time.time(); res=solve(sc, time_limit_s=30); dt=time.time()-t0
    nw,_=nowait_schedule(sc)
    ev=(nw-res['carbon'])/nw*100 if (nw and res['carbon'] is not None) else float('nan')
    print("  n=%3d T=%3d  %-10s %6.2fs  exact=%s  EVPI=%6.2f%%  wait=%s" %
          (n_jobs,T,res['carbon_status'],dt,res['exact'],ev,res['total_wait']))
