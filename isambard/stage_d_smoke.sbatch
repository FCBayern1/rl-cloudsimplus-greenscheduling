#!/bin/bash
# Equivalence smoke on one Isambard-AI aip2 node: does the Stage D chain behave here as it
# does on the workstation? Not an effect measurement; it checks CUDA, the four-line launch
# with one GPU each, the init-hash equality, one short training round, and one deployment
# evaluation row with every field the health gate reads.
#
#   sbatch isambard/stage_d_smoke.sbatch
#SBATCH --job-name=stage_d_smoke
#SBATCH --account=brics.u6tx
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -uo pipefail

BASE=$PROJECTDIR/joshua
REPO=$BASE/rl-cloudsimplus-greenscheduling
source $BASE/miniforge3/etc/profile.d/conda.sh
conda activate sd
export TMPDIR=$SCRATCHDIR/tmp RAY_TMPDIR=$SCRATCHDIR/ray
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
export RAY_LIMIT_CPUS=16 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0
cd "$REPO"

echo "== node =="; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

echo "== 8000-step training of two lines, one GPU each, init hashes must match =="
OUT=$SCRATCHDIR/smoke
rm -rf "$OUT"; mkdir -p "$OUT"
for pair in "NV 0" "V 1"; do
  set -- $pair
  CUDA_VISIBLE_DEVICES=$2 python drl-manager/entrypoint_rlmodule_gtrxl.py \
    --config g1/compressed_timecap_s2/config_stage_d_longrun.yml \
    --experiment sd_$1_s2_r48_w72_c3_n35 --total-timesteps 8000 --num-workers 0 \
    --seed 20260904 --no-wandb --output-dir "$OUT/$1" > "$OUT/$1.log" 2>&1 &
done
wait
python - <<PY
import hashlib, glob, os
def h(d):
    fs = sorted(p for p in glob.glob(os.path.join(d, "learner_group/learner/rl_module/**/*"), recursive=True)
                if os.path.isfile(p) and "state" in os.path.basename(p))
    x = hashlib.sha256()
    for p in fs: x.update(open(p, "rb").read())
    return x.hexdigest()[:12] if fs else None
a, b = h("$OUT/NV/checkpoint_init"), h("$OUT/V/checkpoint_init")
print("init hashes NV", a, "V", b, "MATCH" if a and a == b else "MISMATCH")
PY
for L in NV V; do
  echo "$L final checkpoints: $(find $OUT/$L -type d -name 'checkpoint_0*' | wc -l), init: $([ -f $OUT/$L/checkpoint_init/INIT_MARKER ] && echo yes || echo no)"
  grep -c "Traceback" "$OUT/$L.log"
done

echo "== one deployment evaluation of the V checkpoint (all health-gate fields must appear) =="
CK=$(find $OUT/V -type d -name 'checkpoint_0*' | sort -V | tail -1)
cd drl-manager
EVAL_CONFIG_PATH=$REPO/g1/compressed_timecap_s2/config_stage_d_eval_judgement.yml \
python -m src.baselines.evaluate --experiment sde_s2_r48_w72_c5_n20_godeye \
  --global rllib --new-api --stochastic --checkpoint "$CK" --local drain \
  --episodes 1 --seed 42 --reset-skip 0 --output "$OUT/eval.csv" > "$OUT/eval.log" 2>&1
python - <<PY
import csv
r = list(csv.DictReader(open("$OUT/eval.csv")))[-1]
keys = ["episode_length","completion_rate_mi","ontime_mi_share","deadline_forced_count","total_carbon_kg",
        "global_reward_sum","global_defer_action_rate","green_episode_offset_rows","ep_carbon_norm_clip_count"]
print({k: r.get(k) for k in keys})
missing = [k for k in keys if r.get(k) in (None, "")]
print("MISSING", missing if missing else "none")
PY
echo "== smoke done =="
