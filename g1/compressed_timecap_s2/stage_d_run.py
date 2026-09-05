"""Stage D health-smoke runner (STAGE_D_PREREG Addendum G), mechanical and testable.

  freeze   hash reader, runner, configs, jar and the source files that decide the run
  train    four lines, one seed, two at a time, on the local GPU
  check    every line has checkpoint_init (INIT_MARKER) and a final checkpoint whose
           result.json reached total_timesteps; both load as RLModules
  jobs     build the 252-job list and assert 252 = 180 final + 72 init
  eval     run the jobs (parallel), skipping finished ones
  verdict  stage_d_health_verdict.py on the results

Usage: python stage_d_run.py all  |  freeze|train|check|jobs|eval|verdict
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRL = os.path.join(REPO, "drl-manager")
PY = os.path.join(DRL, ".venv/bin/python")
# Stage D health smoke by default; the D' development smoke passes its own configs and a
# suffix (STAGE_D_CONFIG / STAGE_D_EVAL_CONFIG / STAGE_D_SUFFIX) so nothing frozen moves.
SUFFIX = os.environ.get("STAGE_D_SUFFIX", "")
CONFIG = os.environ.get("STAGE_D_CONFIG") or os.path.join(HERE, "config_stage_d_physical.yml")
EVAL_CONFIG = os.environ.get("STAGE_D_EVAL_CONFIG") or os.path.join(HERE, "config_stage_d_eval.yml")
JAR = os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar")
LOGS = os.path.join(DRL, f"logs/stage_d{SUFFIX}")
RESULTS = os.path.join(DRL, f"results/stage_d{SUFFIX}")
SEED = int(os.environ.get("STAGE_D_SEED", "20260903"))
STEPS = int(os.environ.get("STAGE_D_STEPS", "56000"))
WORKERS = int(os.environ.get("STAGE_D_EVAL_WORKERS", "6"))
LINES = ("NV", "V", "NE", "E")
CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
KS = (26, 34, 42)
TIERS = {"NV": ("hollow",), "NE": ("hollow",),
         "V": ("godeye", "calibrated_shrink_v1", "shuffle", "anti"),
         "E": ("godeye", "calibrated_shrink_v1", "shuffle", "anti")}
CLEAN = {"NV": "hollow", "NE": "hollow", "V": "godeye", "E": "godeye"}
EXPECTED_FINAL, EXPECTED_INIT = 180, 72
FROZEN_SOURCES = [
    "g1/compressed_timecap_s2/stage_d_run.py", "g1/compressed_timecap_s2/stage_d_health_verdict.py",
    "g1/compressed_timecap_s2/gen_stage_d.py",
    os.path.relpath(CONFIG, REPO), os.path.relpath(EVAL_CONFIG, REPO),
    os.path.relpath(CONFIG, REPO).replace("config_stage_d", "stage_d_manifest").replace(".yml", ".json"),
    "drl-manager/src/baselines/evaluate.py", "drl-manager/src/learners/crd_q_loss.py",
    "drl-manager/src/training/train_rlmodule_gtrxl.py", "drl-manager/src/callbacks/init_checkpoint_callback.py",
    "drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py",
    "drl-manager/src/prediction/perturbed_godeye_provider.py", "drl-manager/src/baselines/forecast_perturb.py",
    "g1/compressed_timecap_s2/timecap_error_audit.json",
]


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def exp_name(line):
    return f"sd_{line}_s2_r48_w72_c3_n35"


def line_dir(line):
    return os.path.join(LOGS, f"{line}_s{SEED}")


# ---------------------------------------------------------------- freeze
def freeze():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    art = {"commit": commit, "worktree_clean": dirty == "", "seed": SEED, "steps": STEPS,
           "jar_sha256": sha(JAR), "sources": {p: sha(os.path.join(REPO, p)) for p in FROZEN_SOURCES},
           "jobs": {"final": EXPECTED_FINAL, "init": EXPECTED_INIT, "total": EXPECTED_FINAL + EXPECTED_INIT}}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "stage_d_freeze.json"), "w") as f:
        f.write(json.dumps(art, sort_keys=True, indent=2))
    return art


# ---------------------------------------------------------------- train
def train_one(line):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [PY, "entrypoint_rlmodule_gtrxl.py", "--config", CONFIG, "--experiment", exp_name(line),
           "--total-timesteps", str(STEPS), "--num-workers", "0", "--seed", str(SEED), "--no-wandb",
           "--output-dir", line_dir(line)]
    with open(os.path.join(LOGS, f"{line}_s{SEED}.log"), "w") as lf:
        rc = subprocess.run(cmd, cwd=DRL, stdout=lf, stderr=subprocess.STDOUT).returncode
    log(f"train {line} exit={rc}")
    return rc


def train():
    for pair in (("NV", "V"), ("NE", "E")):
        log(f"training {pair[0]} + {pair[1]}")
        with ThreadPoolExecutor(max_workers=2) as ex:
            rcs = list(ex.map(train_one, pair))
        if any(rcs):
            raise RuntimeError(f"training failed: {dict(zip(pair, rcs))}")


# ---------------------------------------------------------------- check
def init_checkpoint(line):
    p = os.path.join(line_dir(line), "checkpoint_init")
    return p if os.path.exists(os.path.join(p, "INIT_MARKER")) else None


def final_checkpoint(line):
    cks = sorted(glob.glob(os.path.join(line_dir(line), "*", "PPO_*", "checkpoint_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    return cks[-1] if cks else None


def steps_reached(line):
    rj = glob.glob(os.path.join(line_dir(line), "*", "PPO_*", "result.json"))
    if not rj:
        return None
    last = None
    for row in open(rj[0]):
        if row.strip():
            last = json.loads(row)
    if last is None:
        return None
    def find(d, suf):
        for k, v in d.items():
            if isinstance(v, dict):
                x = find(v, suf)
                if x is not None:
                    return x
            elif k.endswith(suf):
                return v
    return find(last, "num_env_steps_sampled_lifetime")


def loads_as_rlmodule(ck):
    """A checkpoint counts as loadable when its multi-RL-module restores in-process."""
    code = ("import sys; from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule; "
            "m = MultiRLModule.from_checkpoint(sys.argv[1]); print(sorted(m.keys()))")
    p = os.path.join(ck, "learner_group", "learner", "rl_module")
    r = subprocess.run([PY, "-c", code, p], cwd=DRL, capture_output=True, text=True, timeout=600)
    return r.returncode == 0, (r.stdout.strip() or r.stderr.strip()[-300:])


def check(load=True):
    out = {}
    for L in LINES:
        ini, fin, st = init_checkpoint(L), final_checkpoint(L), steps_reached(L)
        row = {"init": ini, "final": fin, "steps": st, "steps_ok": st is not None and st >= STEPS}
        if load:
            row["init_loads"], row["init_modules"] = loads_as_rlmodule(ini) if ini else (False, "missing")
            row["final_loads"], row["final_modules"] = loads_as_rlmodule(fin) if fin else (False, "missing")
        out[L] = row
    ok = all(r["init"] and r["final"] and r["steps_ok"] and r.get("init_loads", True) and r.get("final_loads", True)
             for r in out.values())
    out["status"] = "CHECKPOINTS_OK" if ok else "CHECKPOINTS_INVALID"
    return out


# ---------------------------------------------------------------- jobs
def build_jobs(cks):
    """cks: {line: {"init": path, "final": path}} -> list of job dicts; asserts 252."""
    jobs = []
    for L in LINES:
        for c in CELLS:
            for t in TIERS[L]:
                for k in KS:
                    jobs.append({"line": L, "tag": "final", "ck": cks[L]["final"], "cell": c, "tier": t, "k": k})
            for k in KS:
                jobs.append({"line": L, "tag": "init", "ck": cks[L]["init"], "cell": c, "tier": CLEAN[L], "k": k})
    n_final = sum(j["tag"] == "final" for j in jobs)
    n_init = sum(j["tag"] == "init" for j in jobs)
    assert (n_final, n_init) == (EXPECTED_FINAL, EXPECTED_INIT), (n_final, n_init)
    assert len(jobs) == EXPECTED_FINAL + EXPECTED_INIT
    return jobs


def job_paths(j):
    d = os.path.join(RESULTS, f"{j['line']}_{j['tag']}")
    os.makedirs(d, exist_ok=True)
    stem = f"{j['cell']}_{j['tier']}_k{j['k']}"
    return os.path.join(d, stem + ".csv"), os.path.join(d, stem + ".log")


REQUIRED_FIELDS = ("completion_rate_mi", "ontime_mi_share", "deadline_forced_count", "total_carbon_kg",
                   "global_reward_sum", "global_defer_action_rate", "ep_carbon_norm_clip_count")


def done(csv_path):
    """A result counts only when every field the health gate reads is present."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return bool(rows) and all(rows[-1].get(k) not in (None, "") for k in REQUIRED_FIELDS)
    except Exception:
        return False


def run_job(j):
    csv_path, log_path = job_paths(j)
    if done(csv_path):
        return "cached"
    env = dict(os.environ)
    env.update({"GATEWAY_LIBS": os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"),
                "EVAL_CONFIG_PATH": EVAL_CONFIG, "PLANNER_EXPECTED_CAP": "640;512;640;512;192",
                "PLANNER_STATIC_TOTAL_W": "0",
                # One BLAS/torch thread per evaluation process: six concurrent GTrXL
                # inferences on eight cores oversubscribed the CPU (37 min per episode
                # against 9 min with four workers) once each process spawned its own pool.
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1"})
    cmd = [PY, "-m", "src.baselines.evaluate", "--experiment", f"sde_{j['cell']}_{j['tier']}",
           "--global", "rllib", "--new-api", "--stochastic", "--checkpoint", j["ck"], "--local", "drain",
           "--episodes", "1", "--seed", "42", "--reset-skip", str(j["k"]), "--output", csv_path]
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=DRL, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            try:
                os.killpg(proc.pid, 15)
                time.sleep(2)
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
    return "ok" if rc == 0 and done(csv_path) else "failed"


def run_eval(jobs):
    counts = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for j, res in zip(jobs, ex.map(run_job, jobs)):
            counts[res] = counts.get(res, 0) + 1
            n = sum(counts.values())
            if n % 20 == 0 or res == "failed":
                log(f"[{n}/{len(jobs)}] {j['line']} {j['tag']} {j['cell']} {j['tier']} k{j['k']}: {res}")
    return counts


# ---------------------------------------------------------------- main
def main(argv):
    phase = argv[1] if len(argv) > 1 else "all"
    if phase in ("freeze", "all"):
        fz = freeze()
        log(f"freeze: commit {fz['commit'][:8]} clean={fz['worktree_clean']} jar {fz['jar_sha256'][:12]}")
        if phase == "all" and not fz["worktree_clean"]:
            raise SystemExit("worktree not clean; refuse to start the smoke")
    if phase in ("train", "all"):
        train()
    if phase in ("check", "all"):
        ck = check()
        log(f"check: {ck['status']} " + " ".join(f"{L}:steps={ck[L]['steps']},init_loads={ck[L].get('init_loads')},final_loads={ck[L].get('final_loads')}" for L in LINES))
        with open(os.path.join(RESULTS, "stage_d_checkpoints.json"), "w") as f:
            f.write(json.dumps(ck, indent=2, default=str))
        if ck["status"] != "CHECKPOINTS_OK":
            raise SystemExit("checkpoints invalid; no evaluation")
    if phase in ("jobs", "eval", "all"):
        ck = check(load=False)
        jobs = build_jobs({L: {"init": ck[L]["init"], "final": ck[L]["final"]} for L in LINES})
        log(f"jobs: {len(jobs)} = {EXPECTED_FINAL} final + {EXPECTED_INIT} init")
        if phase != "jobs":
            log(f"eval: {run_eval(jobs)}")
    if phase in ("verdict", "all"):
        r = subprocess.run([PY, os.path.join(HERE, "stage_d_health_verdict.py"), RESULTS, LOGS, RESULTS],
                           cwd=DRL, capture_output=True, text=True)
        log("verdict: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]))
        v = json.load(open(os.path.join(RESULTS, "stage_d_health_verdict.json")))
        log(f"VERDICT {v['verdict']}")


if __name__ == "__main__":
    main(sys.argv)
