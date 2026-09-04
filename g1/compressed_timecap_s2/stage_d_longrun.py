"""Stage D long-run runner (STAGE_D_LONGRUN_PREREG, FROZEN with §8), mechanical and testable.

Per seed, in order: freeze (hashes, versions, hardware, disk gate) -> train N_V+V then N_E+E
(two at a time; after both init checkpoints exist their learner RLModule weight hashes must be
equal, N_V=V and N_E=E, or both are killed) -> check (init + final checkpoints load, steps
reached, finite returns) -> eval on the six judgement windows (360 final + 144 init rows) ->
crd stats. After all seeds: verdict (stage_d_longrun_verdict.py). The certified windows are
a separate, secondary phase run after the main verdict exists.

During training this runner prints only liveness, exit codes, step counts, disk and NaN.

Usage: python stage_d_longrun.py all | seed <S> <freeze|train|check|eval|crd> | verdict | certified
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRL = os.path.join(REPO, "drl-manager")
# The workstation's venv by default; a cluster passes its own interpreter (conda env on
# Isambard) through STAGE_D_PYTHON so the frozen source needs no edit at job time.
PY = os.environ.get("STAGE_D_PYTHON") or os.path.join(DRL, ".venv/bin/python")
CONFIG = os.path.join(HERE, "config_stage_d_longrun.yml")
EVAL_JUDGEMENT = os.path.join(HERE, "config_stage_d_eval_judgement.yml")
EVAL_CERTIFIED = os.path.join(HERE, "config_stage_d_eval.yml")
WINDOWS = os.path.join(HERE, "stage_a_out", "stage_d_windows.json")
JAR = os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar")
LOGS = os.path.join(DRL, "logs/stage_d_longrun")
RESULTS = os.path.join(DRL, "results/stage_d_longrun")
# Workstation defaults; a cluster exports its own before calling this runner, and the
# exported value wins (hard-coding the workstation path made every Isambard job die in the
# freeze step with PermissionError on /home/joshua).
RAY_TMPDIR = os.environ.get("RAY_TMPDIR") or "/home/joshua/rt"
SEEDS = (20260904, 20260905, 20260906, 20260907, 20260908)
STEPS = 400_000
WORKERS = int(os.environ.get("STAGE_D_EVAL_WORKERS", "6"))
DISK_MIN_GB = 50
LINES = ("NV", "V", "NE", "E")
PAIRS = (("NV", "V"), ("NE", "E"))
CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
TIERS = {"NV": ("hollow",), "NE": ("hollow",),
         "V": ("godeye", "calibrated_shrink_v1", "shuffle", "anti"),
         "E": ("godeye", "calibrated_shrink_v1", "shuffle", "anti")}
CLEAN = {"NV": "hollow", "NE": "hollow", "V": "godeye", "E": "godeye"}
CERTIFIED_KS = (26, 34, 42)
EXPECTED_MAIN, EXPECTED_INIT = 360, 144
FROZEN_SOURCES = [
    "g1/compressed_timecap_s2/stage_d_longrun.py", "g1/compressed_timecap_s2/stage_d_longrun_verdict.py",
    "g1/compressed_timecap_s2/gen_stage_d.py", "g1/compressed_timecap_s2/config_stage_d_longrun.yml",
    "g1/compressed_timecap_s2/config_stage_d_eval_judgement.yml", "g1/compressed_timecap_s2/config_stage_d_eval.yml",
    "g1/compressed_timecap_s2/stage_a_out/stage_d_windows.json", "g1/compressed_timecap_s2/timecap_error_audit.json",
    "drl-manager/src/baselines/evaluate.py", "drl-manager/src/learners/crd_q_loss.py",
    "drl-manager/src/training/train_rlmodule_gtrxl.py", "drl-manager/src/callbacks/init_checkpoint_callback.py",
    "drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py",
    "drl-manager/src/prediction/perturbed_godeye_provider.py", "drl-manager/src/baselines/forecast_perturb.py",
    "drl-manager/src/models/rlmodule_gtrxl_ensemble.py",
]
# TMPDIR: Ray Tune writes every checkpoint to Python's temp dir before persisting it to
# storage_path; with /tmp full that write raised ENOSPC at 320k of seed 20260904 (long-run
# addendum A). Everything temporary now lives on the large /home partition.
TMPDIR = os.environ.get("TMPDIR") or os.path.join(RAY_TMPDIR, "tmp")
# The filesystem the run writes its outputs to; the disk gate checks this one and TMPDIR's.
DATA_PATH = os.environ.get("STAGE_D_DATA_PATH") or os.path.dirname(os.path.abspath(__file__))
BASE_ENV = {"PYTHONHASHSEED": "0", "RAY_TMPDIR": RAY_TMPDIR, "TMPDIR": TMPDIR,
            "GATEWAY_LIBS": os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"),
            "PLANNER_EXPECTED_CAP": "640;512;640;512;192", "PLANNER_STATIC_TOTAL_W": "0"}
EVAL_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def env_for(extra=None):
    e = dict(os.environ)
    e.update(BASE_ENV)
    if extra:
        e.update(extra)
    return e


def exp_name(line):
    return f"sd_{line}_s2_r48_w72_c3_n35"


def line_dir(seed, line):
    return os.path.join(LOGS, f"{line}_s{seed}")


def seed_results(seed):
    return os.path.join(RESULTS, f"seed_{seed}")


def judgement_offsets():
    return [int(w["offset"]) for w in json.load(open(WINDOWS))["eval_windows"]]


# ---------------------------------------------------------------- gates
def disk_free_gb(path="/home"):
    return shutil.disk_usage(path).free / 1e9


def disk_gate(min_gb=DISK_MIN_GB, path=None, tmp_min_gb=10, tmp_path=None):
    """Hard gate on the partition holding the outputs and on the one holding TMPDIR."""
    path = path or DATA_PATH
    tmp_path = tmp_path or (TMPDIR if os.path.isdir(TMPDIR) else "/tmp")
    free = disk_free_gb(path)
    if free < min_gb:
        raise SystemExit(f"disk gate: {free:.1f} GB free on {path} < {min_gb} GB; refusing to continue")
    tfree = disk_free_gb(tmp_path)
    if tfree < tmp_min_gb:
        raise SystemExit(f"disk gate: {tfree:.1f} GB free on {tmp_path} < {tmp_min_gb} GB; refusing to continue")
    return free


def init_hash(ck_dir):
    """SHA256 over the learner RLModule state files (sorted) of a checkpoint directory."""
    root = os.path.join(ck_dir, "learner_group", "learner", "rl_module")
    files = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if os.path.isfile(p) and "state" in os.path.basename(p))
    h = hashlib.sha256()
    for p in files:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest() if files else None


# ---------------------------------------------------------------- freeze
def freeze_seed(seed):
    os.makedirs(RAY_TMPDIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)
    os.makedirs(seed_results(seed), exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    ver = subprocess.run([PY, "-c", "import torch, ray; print(torch.__version__, torch.version.cuda, ray.__version__)"],
                         cwd=DRL, capture_output=True, text=True).stdout.strip().split()
    art = {"seed": seed, "commit": commit, "worktree_clean": dirty == "", "steps": STEPS,
           "gpu": gpu, "torch": ver[0] if ver else None, "cuda": ver[1] if len(ver) > 1 else None,
           "ray": ver[2] if len(ver) > 2 else None, "python_hash_seed": BASE_ENV["PYTHONHASHSEED"],
           "ray_tmpdir": RAY_TMPDIR, "tmpdir": TMPDIR, "data_path": DATA_PATH,
           "disk_free_gb": round(disk_free_gb(DATA_PATH), 1),
           "tmp_free_gb": round(disk_free_gb(TMPDIR if os.path.isdir(TMPDIR) else "/tmp"), 1),
           "jar_sha256": sha(JAR), "sources": {p: sha(os.path.join(REPO, p)) for p in FROZEN_SOURCES},
           "judgement_offsets": judgement_offsets(),
           "jobs": {"main": EXPECTED_MAIN, "init": EXPECTED_INIT}}
    with open(os.path.join(seed_results(seed), "freeze.json"), "w") as f:
        f.write(json.dumps(art, sort_keys=True, indent=2))
    return art


# ---------------------------------------------------------------- train
def launch_train(seed, line):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [PY, "entrypoint_rlmodule_gtrxl.py", "--config", CONFIG, "--experiment", exp_name(line),
           "--total-timesteps", str(STEPS), "--num-workers", "0", "--seed", str(seed), "--no-wandb",
           "--output-dir", line_dir(seed, line)]
    extra = {}
    # One GPU per line when the node has one per line (STAGE_D_GPU_MAP="NV:0,V:1,NE:2,E:3").
    gmap = os.environ.get("STAGE_D_GPU_MAP", "")
    if gmap:
        m = dict(kv.split(":") for kv in gmap.split(",") if ":" in kv)
        if line in m:
            extra["CUDA_VISIBLE_DEVICES"] = m[line]
    lf = open(os.path.join(LOGS, f"{line}_s{seed}.log"), "w")
    return subprocess.Popen(cmd, cwd=DRL, env=env_for(extra), stdout=lf, stderr=subprocess.STDOUT,
                            start_new_session=True), lf


def wait_init(seed, line, timeout_s=1800):
    marker = os.path.join(line_dir(seed, line), "checkpoint_init", "INIT_MARKER")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(marker):
            time.sleep(5)               # let the checkpoint writer finish
            return os.path.dirname(marker)
        time.sleep(5)
    return None


def kill(proc):
    try:
        os.killpg(proc.pid, 15)
        time.sleep(3)
        os.killpg(proc.pid, 9)
    except ProcessLookupError:
        pass


def train_group(seed, lines):
    """Train `lines` concurrently, one process each, after checking that the paired
    initialisations match (N_V = V, N_E = E). A mismatch kills the whole group."""
    disk_gate()
    procs = {L: launch_train(seed, L) for L in lines}
    inits = {L: wait_init(seed, L) for L in lines}
    hashes = {L: (init_hash(inits[L]) if inits[L] else None) for L in lines}
    log(f"seed {seed} group {tuple(lines)}: init hashes " + " ".join(f"{L}={str(h)[:12]}" for L, h in hashes.items()))
    bad = [p for p in PAIRS if set(p) <= set(lines)
           and (hashes[p[0]] is None or hashes[p[0]] != hashes[p[1]])]
    if bad or any(hashes[L] is None for L in lines):
        for p, _ in procs.values():
            kill(p)
        raise SystemExit(f"init weight hashes differ or missing for {bad or lines} (seed {seed}); training killed")
    rcs = {}
    for L, (p, lf) in procs.items():
        rcs[L] = p.wait()
        lf.close()
        log(f"seed {seed} train {L} exit={rcs[L]} disk_free={disk_free_gb(DATA_PATH):.1f}GB")
    if any(rcs.values()):
        raise SystemExit(f"training failed: {rcs}")
    with open(os.path.join(seed_results(seed), f"init_hashes_{'_'.join(lines)}.json"), "w") as f:
        f.write(json.dumps(hashes, indent=2))


def train_groups():
    """How many lines run at once: pairs on a single-GPU workstation, all four on a node
    with four GPUs (STAGE_D_PARALLEL_LINES=4). Scientifically identical: each line is an
    independent seeded process; only wall-clock and GPU placement change."""
    n = int(os.environ.get("STAGE_D_PARALLEL_LINES", "2"))
    return [list(LINES)] if n >= 4 else [list(p) for p in PAIRS]


def train_seed(seed):
    for group in train_groups():
        train_group(seed, group)


def train_line(seed, line):
    """One line, one process, for a cluster that schedules a job per GPU. The paired
    init-hash check cannot run here (the partner is a different job), so it is enforced
    by `check_init_pairs` before that seed's evaluation."""
    disk_gate()
    p, lf = launch_train(seed, line)
    rc = p.wait()
    lf.close()
    log(f"seed {seed} train {line} exit={rc} disk_free={disk_free_gb(DATA_PATH):.1f}GB")
    if rc:
        raise SystemExit(f"training failed: {line} rc={rc}")


def check_init_pairs(seed):
    """N_V = V and N_E = E over the four init checkpoints of a seed. Run before the
    seed's evaluation when the lines were trained as separate jobs."""
    hashes = {L: (init_hash(init_checkpoint(seed, L)) if init_checkpoint(seed, L) else None) for L in LINES}
    log(f"seed {seed} init hashes " + " ".join(f"{L}={str(h)[:12]}" for L, h in hashes.items()))
    bad = [p for p in PAIRS if hashes[p[0]] is None or hashes[p[0]] != hashes[p[1]]]
    with open(os.path.join(seed_results(seed), "init_hashes.json"), "w") as f:
        f.write(json.dumps({"hashes": hashes, "pairs_ok": not bad}, indent=2))
    if bad:
        raise SystemExit(f"init weight hashes differ or missing for {bad} (seed {seed})")
    return hashes


# ---------------------------------------------------------------- check
def final_checkpoint(seed, line):
    cks = sorted(glob.glob(os.path.join(line_dir(seed, line), "*", "PPO_*", "checkpoint_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    return cks[-1] if cks else None


def init_checkpoint(seed, line):
    p = os.path.join(line_dir(seed, line), "checkpoint_init")
    return p if os.path.exists(os.path.join(p, "INIT_MARKER")) else None


def last_result(seed, line):
    rj = glob.glob(os.path.join(line_dir(seed, line), "**", "result.json"), recursive=True)
    if not rj:
        return None, 0
    last, n = None, 0
    for row in open(rj[0]):
        if row.strip():
            last = json.loads(row); n += 1
    return last, n


def _find(d, suf):
    for k, v in d.items():
        if isinstance(v, dict):
            x = _find(v, suf)
            if x is not None:
                return x
        elif k.endswith(suf):
            return v


def loads(ck):
    code = ("import sys; from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule; "
            "MultiRLModule.from_checkpoint(sys.argv[1]); print('ok')")
    p = os.path.join(ck, "learner_group", "learner", "rl_module")
    r = subprocess.run([PY, "-c", code, p], cwd=DRL, env=env_for(), capture_output=True, text=True, timeout=900)
    return r.returncode == 0


def check_seed(seed, load=True):
    out = {}
    for L in LINES:
        ini, fin = init_checkpoint(seed, L), final_checkpoint(seed, L)
        last, n_iter = last_result(seed, L)
        steps = _find(last, "num_env_steps_sampled_lifetime") if last else None
        ret = _find(last, "episode_return_mean") if last else None
        finite = ret is not None and ret == ret and abs(ret) != float("inf")
        row = {"init": ini, "final": fin, "iterations": n_iter, "steps": steps,
               "steps_ok": steps is not None and steps >= STEPS, "return_finite": finite}
        if load:
            row["init_loads"] = loads(ini) if ini else False
            row["final_loads"] = loads(fin) if fin else False
        out[L] = row
    ok = all(r["init"] and r["final"] and r["steps_ok"] and r["return_finite"]
             and r.get("init_loads", True) and r.get("final_loads", True) for r in out.values())
    out["status"] = "CHECKPOINTS_OK" if ok else "CHECKPOINTS_INVALID"
    with open(os.path.join(seed_results(seed), "checkpoints.json"), "w") as f:
        f.write(json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------- eval
def build_jobs(seed, cks, ks, tag_prefix=""):
    jobs = []
    for L in LINES:
        for c in CELLS:
            for t in TIERS[L]:
                for k in ks:
                    jobs.append({"seed": seed, "line": L, "tag": "final", "ck": cks[L]["final"], "cell": c, "tier": t, "k": k})
            for k in ks:
                jobs.append({"seed": seed, "line": L, "tag": "init", "ck": cks[L]["init"], "cell": c, "tier": CLEAN[L], "k": k})
    return jobs


def assert_main_counts(jobs):
    n_final = sum(j["tag"] == "final" for j in jobs)
    n_init = sum(j["tag"] == "init" for j in jobs)
    assert (n_final, n_init) == (EXPECTED_MAIN, EXPECTED_INIT), (n_final, n_init)
    return n_final, n_init


def job_paths(j, subdir=""):
    d = os.path.join(seed_results(j["seed"]), subdir, f"{j['line']}_{j['tag']}")
    os.makedirs(d, exist_ok=True)
    stem = f"{j['cell']}_{j['tier']}_k{j['k']}"
    return os.path.join(d, stem + ".csv"), os.path.join(d, stem + ".log")


REQUIRED_FIELDS = ("completion_rate_mi", "ontime_mi_share", "deadline_forced_count", "total_carbon_kg",
                   "global_reward_sum", "global_defer_action_rate", "green_episode_offset_rows")


def done(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return bool(rows) and all(rows[-1].get(k) not in (None, "") for k in REQUIRED_FIELDS)
    except Exception:
        return False


def run_job(j, config, subdir=""):
    csv_path, log_path = job_paths(j, subdir)
    if done(csv_path):
        return "cached"
    cmd = [PY, "-m", "src.baselines.evaluate", "--experiment", f"sde_{j['cell']}_{j['tier']}",
           "--global", "rllib", "--new-api", "--stochastic", "--checkpoint", j["ck"], "--local", "drain",
           "--episodes", "1", "--seed", "42", "--reset-skip", str(j["k"]), "--output", csv_path]
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=DRL, env=env_for({**EVAL_ENV, "EVAL_CONFIG_PATH": config}),
                                stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            kill(proc)
    return "ok" if rc == 0 and done(csv_path) else "failed"


def run_jobs(jobs, config, subdir=""):
    counts = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for j, res in zip(jobs, ex.map(lambda jj: run_job(jj, config, subdir), jobs)):
            counts[res] = counts.get(res, 0) + 1
            n = sum(counts.values())
            if n % 36 == 0 or res == "failed":
                log(f"[{n}/{len(jobs)}] seed {j['seed']} {j['line']} {j['tag']} {j['cell']} {j['tier']} k{j['k']}: {res}")
    return counts


def eval_seed(seed):
    disk_gate()
    ck = check_seed(seed, load=False)
    if ck["status"] != "CHECKPOINTS_OK":
        raise SystemExit("checkpoints invalid; no evaluation")
    offs = judgement_offsets()
    jobs = build_jobs(seed, {L: {"init": ck[L]["init"], "final": ck[L]["final"]} for L in LINES}, range(len(offs)))
    assert_main_counts(jobs)
    log(f"seed {seed} eval: {len(jobs)} = {EXPECTED_MAIN} main + {EXPECTED_INIT} init on the judgement windows")
    counts = run_jobs(jobs, EVAL_JUDGEMENT)
    log(f"seed {seed} eval done: {counts}")
    # every row must have replayed the registered offset for its k
    bad = []
    for j in jobs:
        p, _ = job_paths(j)
        if done(p):
            r = list(csv.DictReader(open(p)))[-1]
            if int(float(r["green_episode_offset_rows"])) != offs[j["k"]]:
                bad.append((j["line"], j["tag"], j["cell"], j["tier"], j["k"], r["green_episode_offset_rows"]))
    log(f"seed {seed} offset check: {len(bad)} mismatches")
    if bad:
        with open(os.path.join(seed_results(seed), "offset_mismatches.json"), "w") as f:
            f.write(json.dumps(bad, indent=2))


def crd_seed(seed):
    out = {}
    for L in ("NE", "E"):
        last, _ = last_result(seed, L)
        if not last:
            continue
        flat = {}

        def walk(d, p=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v, p + k + "/")
                else:
                    flat[p + k] = v
        walk(last)
        pick = lambda s: next((v for k, v in flat.items() if k.endswith(s)), None)  # noqa: E731
        out[L] = {s: pick("crd/" + s) for s in ("dr_mean", "dr_std", "dq_std", "rho_routing_mean", "rho_routing_std",
                                                 "rho_forecast_mean", "rho_scheduling_mean", "reweight_w_std", "sigma2_tot_mean", "c_t_mean")}
    with open(os.path.join(seed_results(seed), "crd_stats.json"), "w") as f:
        f.write(json.dumps(out, indent=2, default=str))
    return out


def certified_seed(seed):
    ck = check_seed(seed, load=False)
    jobs = [j for j in build_jobs(seed, {L: {"init": ck[L]["init"], "final": ck[L]["final"]} for L in LINES},
                                  CERTIFIED_KS) if j["tag"] == "final"]
    log(f"seed {seed} certified (secondary): {len(jobs)} rows")
    log(f"seed {seed} certified done: {run_jobs(jobs, EVAL_CERTIFIED, subdir='certified')}")


def verdict():
    r = subprocess.run([PY, os.path.join(HERE, "stage_d_longrun_verdict.py"), RESULTS], cwd=DRL,
                       capture_output=True, text=True)
    log("verdict: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:]))
    v = json.load(open(os.path.join(RESULTS, "stage_d_longrun_verdict.json")))
    log(f"VERDICT {v['verdict']}")
    return v


def main(argv):
    phase = argv[1] if len(argv) > 1 else "all"
    if phase == "all":
        for s in SEEDS:
            fz = freeze_seed(s)
            if not fz["worktree_clean"]:
                raise SystemExit("worktree not clean; refuse to start")
            log(f"seed {s} frozen: commit {fz['commit'][:8]} gpu={fz['gpu']} disk={fz['disk_free_gb']}GB")
            train_seed(s)
            ck = check_seed(s)
            log(f"seed {s} check: {ck['status']}")
            if ck["status"] != "CHECKPOINTS_OK":
                raise SystemExit("checkpoints invalid")
            eval_seed(s)
            crd_seed(s)
        verdict()
        for s in SEEDS:
            certified_seed(s)
        log("long run done")
    elif phase == "seed":
        s, sub = int(argv[2]), argv[3]
        if sub == "train_line":
            train_line(s, argv[4])
        elif sub == "check_init_pairs":
            check_init_pairs(s)
        else:
            {"freeze": freeze_seed, "train": train_seed, "check": check_seed, "eval": eval_seed,
             "crd": crd_seed, "certified": certified_seed}[sub](s)
    elif phase == "verdict":
        verdict()
    elif phase == "certified":
        for s in SEEDS:
            certified_seed(s)


if __name__ == "__main__":
    main(sys.argv)
