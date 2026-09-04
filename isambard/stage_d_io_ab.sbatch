#!/bin/bash
# Is the three-times-slower training on Isambard caused by Lustre I/O or by the CPU?
# A/B: the same 8000-step training twice in parallel, one writing to $SCRATCHDIR (Lustre),
# one to node-local storage. Same node, same GPU model, one GPU each.
#SBATCH --job-name=stage_d_io_ab
#SBATCH --account=brics.u6tx
#SBATCH --partition=workq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -uo pipefail

BASE=$PROJECTDIR/joshua
REPO=$BASE/rl-cloudsimplus-greenscheduling
source $BASE/miniforge3/etc/profile.d/conda.sh
conda activate sd
export RAY_LIMIT_CPUS=16 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib

LOCAL=${LOCALDIR:-/local/user/$(id -u)}
echo "== node $(hostname); local storage $LOCAL =="
mkdir -p "$LOCAL/io_ab" 2>/dev/null && echo "local writable: $(df -h $LOCAL | tail -1)" || { echo "LOCAL NOT WRITABLE, falling back to /dev/shm"; LOCAL=/dev/shm/$USER; mkdir -p "$LOCAL/io_ab"; }
df -h "$SCRATCHDIR" | tail -1

run(){  # name, output root, gpu, tmp root
  local NAME=$1 ROOT=$2 GPU=$3 TMP=$4
  mkdir -p "$ROOT/$NAME" "$TMP"
  TMPDIR=$TMP RAY_TMPDIR=$TMP CUDA_VISIBLE_DEVICES=$GPU \
  python "$REPO/drl-manager/entrypoint_rlmodule_gtrxl.py" \
    --config "$REPO/g1/compressed_timecap_s2/config_stage_d_longrun.yml" \
    --experiment sd_NV_s2_r48_w72_c3_n35 --total-timesteps 8000 --num-workers 0 \
    --seed 20260904 --no-wandb --output-dir "$ROOT/$NAME" > "$ROOT/$NAME.log" 2>&1
}

echo "== running both variants in parallel =="
run lustre "$SCRATCHDIR/io_ab" 0 "$SCRATCHDIR/io_ab/tmp" &
run local  "$LOCAL/io_ab"      1 "$LOCAL/io_ab/tmp" &
wait

python - <<PY
import glob, json
for name, root in (("lustre", "$SCRATCHDIR/io_ab"), ("local", "$LOCAL/io_ab")):
    f = glob.glob(f"{root}/{name}/*/PPO_*/result.json")
    if not f:
        print(name, "no result.json"); continue
    rows = [json.loads(l) for l in open(f[0]) if l.strip()]
    r = rows[-1]
    def find(d, s):
        for k, v in d.items():
            if isinstance(v, dict):
                x = find(v, s)
                if x is not None: return x
            elif k.endswith(s): return v
    print(f"{name:7s} iters {len(rows)} steps {find(r,'num_env_steps_sampled_lifetime')} "
          f"time_total_s {float(r.get('time_total_s',0)):.0f} "
          f"env_step_s {float(find(r,'env_runner_sampling_timer') or 0):.2f} "
          f"learn_s {float(find(r,'learner_update_timer') or 0):.2f}")
PY
echo "== gateway log volume written =="
du -sh "$SCRATCHDIR/io_ab" "$LOCAL/io_ab" 2>/dev/null
echo "== io ab done =="
