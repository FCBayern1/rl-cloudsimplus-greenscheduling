import os, numpy as np, yaml
os.environ["EVAL_CONFIG_PATH"]="/home/joshua/rl-cloudsimplus-greenscheduling/config_C.yml"
cfg=yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
exp=cfg["experiment_multi_5dc_carbon_v2_deferrable_gdpd"]
exp["cloudlet_trace_file"]="traces/solar_longjob_n1200.csv"
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
env=HierarchicalMultiDCEnv(exp)
obs,_=env.reset(seed=42)
g0=(obs.get('global') if isinstance(obs,dict) and 'global' in obs else obs)
print("GREEN-related obs keys:", [k for k in g0 if 'green' in k.lower() or 'future' in k.lower()])
def grab(o):
    g=o.get('global') if isinstance(o,dict) and 'global' in o else o
    return g.get('dc_current_green_power_w', g.get('dc_green_ratio'))
for step in range(2300):
    a=env.action_space.sample()
    obs,_,term,trunc,_=env.step(a)
    if step in (50,300,600,1000,1400,1800,2200):
        print(f"  t≈{step}: green/ratio per DC = {np.round(np.asarray(grab(obs)),3)}")
    if term or trunc: print("ep end",step); break
env.close()
