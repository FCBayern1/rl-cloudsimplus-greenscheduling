"""Read-only checkpoint diagnostics. Never trains, modifies checkpoints or frozen verdicts.

Both replay modes explicitly disable network dropout. 'legacy' reproduces only
the historical state-free call, NOT its active dropout; recurrent carries STATE_OUT.
Artifacts are diagnostic, not replacements for registered experiment results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'drl-manager'))
import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module_path(ck):
    return Path(ck).resolve() / 'learner_group/learner/rl_module/global_policy'


def load_module(ck):
    from ray.rllib.core.rl_module.rl_module import RLModule
    m = RLModule.from_checkpoint(str(module_path(ck)))
    m.to('cpu').eval()
    assert m.offset_mode and m.cover_prior_fixed and float(m.cover_gain) == 20
    return m


def batch_obs(obs):
    return {'obs': {'observation': {k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
                                    for k, v in obs.items()}}}


def distribution_stats(logits, cover, allowed, valid):
    z = logits.reshape(*cover.shape).double()[valid]
    cv = cover.double()[valid]
    al = allowed[valid] >= .5
    if not len(z):
        return None
    prior_z = torch.where(al, 20 * cv, torch.full_like(cv, -1e9))
    lp, lq = z.log_softmax(-1), prior_z.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    action, rule = z.argmax(-1), prior_z.argmax(-1)
    residual = (z - 20 * cv)[al]
    return dict(actions=action.tolist(), rules=rule.tolist(),
                entropy=(-(p * lp).sum(-1)).tolist(),
                kl_prior=(p * (lp-lq)).sum(-1).tolist(),
                prior_action_prob=p.gather(1, rule[:, None]).flatten().tolist(),
                residual_min=float(residual.min()), residual_max=float(residual.max()),
                logp=lp.detach().numpy())


class DiagnosticPolicy:
    def __init__(self, module, mode):
        self.module, self.mode = module, mode
        self.reset()

    def reset(self):
        self.state = None
        self.rows = []
        self.values = []
        self.step = 0

    def schedule(self, observation):
        obs = observation.get('raw_global_obs', observation)
        batch = batch_obs(obs)
        if self.mode == 'recurrent' and self.state is not None:
            batch['state_in'] = self.state
        with torch.no_grad():
            logits, values, state, = self.module._forward_pass(batch)
        if self.mode == 'recurrent':
            self.state = {k: v.detach() for k, v in state.items()}
        self.values.append(float(values.flatten()[-1]))
        cover = torch.as_tensor(obs['cand_green_cover'])
        allowed = torch.as_tensor(obs['batch_cloudlet_offset_allowed'])
        valid = torch.as_tensor(obs['batch_cloudlet_mi']) > 0
        stats = distribution_stats(logits, cover, allowed, valid)
        if stats is not None:
            stats.pop('logp')
            self.rows.append({'step': self.step, 'value': float(values.flatten()[-1]), **stats})
        self.step += 1
        return logits.reshape(self.module.num_batch_slots, -1).argmax(-1).tolist()


def summarize(rows):
    actions = np.array([x for r in rows for x in r['actions']])
    rules = np.array([x for r in rows for x in r['rules']])
    return dict(decisions=len(actions), agreement=float(np.mean(actions == rules)),
                mean_offset=float(np.mean(actions % 73)),
                mean_absolute_offset_difference=float(np.mean(abs(actions % 73-rules % 73))),
                site_histogram=np.bincount(actions // 73, minlength=5).tolist(),
                **{k: float(np.mean([x for r in rows for x in r[k]]))
                   for k in ('entropy', 'kl_prior', 'prior_action_prob')},
                residual_min=min(r['residual_min'] for r in rows),
                residual_max=max(r['residual_max'] for r in rows),
                value_mean=float(np.mean([r['value'] for r in rows])))


def replay(args):
    # The evaluation environment, weather, local drain, candidate mask and reward
    # are the existing implementation. Only the isolated diagnostic actor is injected.
    from src.baselines import evaluate as ev
    from src.baselines.global_schedulers import GLOBAL_SCHEDULERS
    from src.baselines.base import GlobalScheduler
    m = load_module(args.checkpoint)
    if args.prior:
        with torch.no_grad():
            for lin in (m.dc_encoder, m.ctx_to_dc, m.offset_head):
                lin.weight.zero_(); lin.bias.zero_()
    policy = DiagnosticPolicy(m, args.mode)
    verified_scheduler=None
    if args.verify_scheduler:
        from types import SimpleNamespace
        from src.baselines.global_schedulers import RLlibNewAPIGlobalScheduler
        assert args.mode=='recurrent'
        verified_scheduler=RLlibNewAPIGlobalScheduler(5,128,
            SimpleNamespace(env_runner=SimpleNamespace(module={'global_policy':m})))
    # Observe, do not change, the environment's transition/reward path.
    trajectory=[]
    base_env=ev.HierarchicalMultiDCEnv
    class RecordedEnvironment(base_env):
        def step(self, action):
            result=super().step(action)
            info=result[4]
            ges=info.get('global_energy_stats', {}) or {}
            trajectory.append(dict(reward=float(result[1]['global']),
                terminated=bool(result[2]),truncated=bool(result[3]),
                **{k:float(v) for k,v in ges.items() if isinstance(v,(int,float)) and
                   (k.startswith('global_reward_term_') or k in ('global_carbon_signal_sum','sla_cost_step'))}))
            return result
    ev.HierarchicalMultiDCEnv=RecordedEnvironment

    class Adapter(GlobalScheduler):
        def schedule(self, obs):
            actions=policy.schedule(obs)
            if verified_scheduler is not None:
                actual=verified_scheduler.schedule(obs.get('raw_global_obs',obs))
                assert actual==actions, 'Production scheduler differs from stepwise diagnostic'
                return actual
            return actions
        def reset(self):
            policy.reset()
            if verified_scheduler is not None:verified_scheduler.reset()

    GLOBAL_SCHEDULERS['investigation_only'] = Adapter
    cfg = ev.load_config('rl2e_full_' + args.tier)
    cfg['py4j_port'] = None
    cfg['gateway_log_dir'] = str(Path(args.out).resolve().parent / 'gateways')
    offset = [21850, 1839, 24859, 28745, 41897, 7934][args.window]
    os.environ.update(ORACLE_OFFSET_ROWS=str(offset), ORACLE_EXPERIMENT='rl2e_full_'+args.tier)
    result = ev.run_evaluation('investigation_only', 'drain', cfg, num_episodes=1,
                              reset_skip=args.window, seed=42,
                              output_csv=args.out+'.csv', verbose=False)
    record = dict(checkpoint=str(Path(args.checkpoint).resolve()), mode=args.mode,
                  prior_override=args.prior,
                  production_scheduler_verified=args.verify_scheduler,
                  tier=args.tier, window=args.window, offset=offset,
                  metrics=result, summary=summarize(policy.rows), decisions=policy.rows,
                  trajectory=trajectory, values=policy.values,
                  discounted_reward={str(g):sum(g**i*r['reward'] for i,r in enumerate(trajectory))
                                     for g in (.99,.999,1.0)})
    write_json(args.out+'.json', record)
    print(json.dumps({**record['summary'], 'carbon': result[0]['total_carbon_kg']}), flush=True)


def fixed(args):
    with np.load(args.corpus) as archive:
        z = {k: archive[k] for k in archive.files}
    cks = sorted(Path(args.root).glob('checkpoint_*'))
    all_records = []
    for ck in cks:
        m = load_module(ck)
        keys = sorted(m.observation_space['observation'].spaces)
        assert not set(keys)-set(z), set(keys)-set(z)
        for mode in ('legacy', 'recurrent'):
            p = DiagnosticPolicy(m, mode)
            for i in range(len(z[keys[0]])):
                p.schedule({k: z[k][i] for k in keys})
            step = 0 if ck.name == 'checkpoint_init' else (int(ck.name.split('_')[-1]) + 1) * 8000
            rec = dict(checkpoint=ck.name, step=step,
                       mode=mode, **summarize(p.rows))
            all_records.append(rec)
            print(json.dumps(rec), flush=True)
        write_json(args.out, dict(corpus=str(Path(args.corpus).resolve()), records=all_records))


def main():
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('replay')
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--mode', choices=['legacy', 'recurrent'], required=True)
    r.add_argument('--tier', choices=['godeye', 'shrink75', 'shrink0'], required=True)
    r.add_argument('--window', type=int, choices=range(6), default=0)
    r.add_argument('--out', required=True)
    r.add_argument('--prior',action='store_true',help='zero learned heads in memory, never save them')
    r.add_argument('--verify-scheduler',action='store_true',help='verify and execute the corrected production adapter')
    f = sp.add_parser('fixed')
    f.add_argument('--root', required=True)
    f.add_argument('--corpus', required=True)
    f.add_argument('--out', required=True)
    args = ap.parse_args()
    {'replay': replay, 'fixed': fixed}[args.cmd](args)


if __name__ == '__main__':
    main()
