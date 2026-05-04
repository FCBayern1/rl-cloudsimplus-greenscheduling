#!/bin/bash
#SBATCH --job-name=timecap_gpu_bench
#SBATCH --output=logs/timecap_gpu_bench_%j.out
#SBATCH --error=logs/timecap_gpu_bench_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

# TimeCAP GPU inference benchmark.
#
# Submits a single-GPU benchmarking job that measures:
#   - End-to-end TimeCAPGodEyeProvider.step_and_get() latency (the path RL pays)
#   - Pure model.forward() latency for batched multi-turbine inputs
#   - Both swept over N = 1, 2, 4, 8, 16, 32, 64, 134 turbines
#
# Usage:
#   sbatch scripts/timecap/benchmark_gpu.sh
#
# Output:
#   logs/timecap_gpu_bench_<jobid>.out — formatted latency tables
#   logs/timecap_gpu_bench_<jobid>.err — any errors / module load output
#
# Override anything via env vars before sbatch (all optional):
#   CHECKPOINT=/path/to/ckpt_best.pth
#   TURBINE_COUNTS=1,4,16,64       # which N to sweep
#   BENCH_ITERS=100                # timed forwards per data point
#   WARMUP_ITERS=10                # warm-up forwards before timing

set -euo pipefail

module load brics/nccl brics/aws-ofi-nccl
source /projects/u6fy/shared_envs/activate_rlcs.sh

mkdir -p logs

REPO_ROOT="/lus/lfs1aip2/projects/u6fy/rl-cloudsimplus-greenscheduling"
cd "${REPO_ROOT}"

# Defaults — override on the sbatch command line if you like
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/drl-manager/timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth}"
TURBINE_COUNTS="${TURBINE_COUNTS:-1,2,4,8,16,32,64,134}"
BENCH_ITERS="${BENCH_ITERS:-100}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"

echo "=========================================================="
echo "  Job ID         : ${SLURM_JOB_ID:-N/A}"
echo "  Node           : $(hostname)"
echo "  GPU(s) visible : ${CUDA_VISIBLE_DEVICES:-?}"
echo "  CPUs per task  : ${SLURM_CPUS_PER_TASK:-?}"
echo "  Checkpoint     : ${CHECKPOINT}"
echo "  Turbine counts : ${TURBINE_COUNTS}"
echo "  Bench iters    : ${BENCH_ITERS}  (warmup ${WARMUP_ITERS})"
echo "=========================================================="
echo ""
nvidia-smi -L || true
echo ""

# Restrict CPU thread pools so the benchmark numbers are deterministic.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u scripts/timecap/benchmark_gpu.py \
    --device "cuda" \
    --checkpoint "${CHECKPOINT}" \
    --turbine-counts "${TURBINE_COUNTS}" \
    --bench-iters "${BENCH_ITERS}" \
    --warmup-iters "${WARMUP_ITERS}"

echo ""
echo "=== done ==="
