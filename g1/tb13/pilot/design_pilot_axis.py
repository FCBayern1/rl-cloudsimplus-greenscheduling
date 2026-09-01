import sys, csv
import numpy as np
sys.path.insert(0,'/home/joshua/rl-cloudsimplus-greenscheduling/g1/tb13')
from exact_oracle import Scenario, solve, nowait_schedule
WD='/home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/simplified'
def ser(t,y=2021):
    return np.array([float(r['power_kw'] or 0) for r in csv.DictReader(open(f'{WD}/Turbine_{t}_{y}.csv'))])
s12,s36,s96=ser(12),ser(36),ser(96)
rng=np.random.default_rng(1)
T=36; off=20000; n=10
green0=np.stack([s12[off:off+T], s36[off:off+T], s96[off:off+T]])
print("axis sweep: demand/green ratio (job dynamic power scaled), 3 DC, n=10, T=36")
print("  dyn_W/PE  mean_green_W  demand/green   status      EVPI%%   wait  p_wait")
for dyn in (2.54, 25, 100, 250, 500, 900):
    # keep the wind shape, scale it to a fixed mean so the ratio is the only axis moving
    green = green0/green0.mean()*400.0
    static=[60.]*3
    a=rng.integers(0,T//3,n); r=rng.integers(2,5,n)
    sc=Scenario(green_w=green, static_w=static, brown_factor=[0.3,0.5,0.7],
                green_factor=[0.02]*3, cap_pes=[8,8,8], arrival=a, runtime=r,
                pes=[2]*n, deadline=[T]*n, dyn_w_per_pe=dyn,
                per_job_wait_max=24, budget_total=6*n)
    res=solve(sc, time_limit_s=30); nw,_=nowait_schedule(sc)
    if nw is None or res['carbon'] is None: print("  %8.1f  infeasible"%dyn); continue
    ev=(nw-res['carbon'])/nw*100
    demand = 2*dyn*n*float(np.mean(r))/T
    pw = sum(1 for i,(d,s) in res['assign'].items() if s>sc.a[i])/n
    print("  %8.1f  %12.1f  %12.3f   %-10s %6.2f  %5d  %5.2f" %
          (dyn, green.mean(), demand/green.mean(), res['carbon_status'], ev,
           res['total_wait'], pw))
