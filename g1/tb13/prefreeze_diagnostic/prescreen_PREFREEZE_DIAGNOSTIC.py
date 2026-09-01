import sys, time
import numpy as np
sys.path.insert(0,'/home/joshua/rl-cloudsimplus-greenscheduling/g1/tb13')
import instance_gen as ig
pool=[int(x) for x in open('/home/joshua/rl-cloudsimplus-greenscheduling/g1/tb13/data_split.txt').read().split('DISCOVERY [')[1].split(']')[0].split(',')]
grid=ig.axes_grid()
RHO_BAND=(0.2, 3.0)          # spans the interior and both shoulders seen in the pilot
CORR=(0.70, 0.95)
t0=time.time(); kept=0; total=0; rhos=[]
for tps in ig.TURBINES_PER_SITE:
    triples=ig.turbine_triples(pool, tps, 6)
    offs=ig.offsets_for(2021, max(ig.HORIZON), 6)
    for ti,(tr,off) in enumerate(zip(triples, offs)):
        for ax in grid:
            if ax['turbines_per_site']!=tps: continue
            total+=1
            a=dict(ax); a['turbines']=tr; a['offset']=off
            try: sc,prov=ig.build_instance(a, seed=0)
            except Exception: continue
            rho=prov['rho_residual']; rhos.append(rho)
            if not (RHO_BAND[0]<=rho<=RHO_BAND[1]): continue
            g=sc.green - sc.static.reshape(-1,1)
            if np.any(g.std(axis=1)<1e-9): continue
            C=np.corrcoef(g)
            pw=[C[i,j] for i in range(3) for j in range(i+1,3)]
            if not all(CORR[0]<=abs(x)<=CORR[1] for x in pw): continue
            gres=np.maximum(g,0.0)
            if (gres.min(axis=0)<=0).mean() in (0.0,1.0): continue
            best=np.argmin(sc.cb.reshape(-1,1)*np.maximum(sc.static.reshape(-1,1)-sc.green,0)+1e-9*np.arange(3).reshape(-1,1), axis=0)
            if len(np.unique(best))<2: pass
            kept+=1
print(f"pre-screen over {total} cells in {time.time()-t0:.1f}s")
r=np.array(rhos)
print(f"  rho_residual  min={r.min():.4f} p10={np.percentile(r,10):.3f} p50={np.percentile(r,50):.3f} p90={np.percentile(r,90):.3f} max={r.max():.1f}")
print(f"  in rho band [{RHO_BAND[0]}, {RHO_BAND[1]}]: {((r>=RHO_BAND[0])&(r<=RHO_BAND[1])).mean()*100:.1f}%")
print(f"  survived all pre-screen gates: {kept} ({100*kept/max(total,1):.1f}%)")
