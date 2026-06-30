#!/bin/bash
# Run ONCE on the Isambard-AI LOGIN node. Builds the aarch64 conda env + the Java gateway jar.
# The x86 .venv from your workstation CANNOT be reused — this rebuilds the stack for ARM/aarch64.
# $PROJECTDIR is PRE-SET by Isambard (= /projects/<PROJECT>); the repo lives under it.
#
#   bash "$PROJECTDIR/rl-cloudsimplus-greenscheduling/isambard/setup_env.sh"
set -euo pipefail
: "${PROJECTDIR:?\$PROJECTDIR should be pre-set by Isambard — run 'echo \$PROJECTDIR' to check}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (parent of isambard/)

# 1) Miniforge for aarch64 (idempotent)
if [ ! -d "$HOME/miniforge3" ]; then
  curl -fsSL -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/bin/activate"

# 2) env: python 3.12 + a JDK (openjdk via conda → no module needed; gradle/Py4J use it)
conda create -y -n rl python=3.12 || true
conda activate rl
conda install -y -c conda-forge "openjdk=21"
export JAVA_HOME="$CONDA_PREFIX"

# 3) PyTorch for aarch64 + CUDA (GH200/H100). If cu124 lacks an aarch64 wheel for 2.11, try the others.
pip install --upgrade pip
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu124 \
 || pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu126 \
 || pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 \
 || { echo "!! pip torch failed on aarch64 — fall back to an NGC Apptainer container (see isambard/README)"; exit 1; }

# 4) Ray/RLlib + the rest (all ship aarch64 manylinux wheels)
# IMPORTANT: pin pyarrow==22 + msgpack==1.1.2. A fresh aarch64 install pulls pyarrow 24 / msgpack 1.2,
# under which ray 2.40.0's episode->batch SEQUENCE CHUNKING silently breaks (stateful GTrXL gets the whole
# episode as one T=2000+ sequence instead of chunks of max_seq_len), crashing the action dist at the learner.
# These exact versions match the (working) workstation env.
pip install "ray[rllib]==2.40.0" "gymnasium==1.0.0" "pettingzoo==1.24.3" "py4j==0.10.9.9" \
            "numpy==2.4.4" "pandas==2.3.3" "scipy==1.17.1" "PyYAML==6.0.3" "wandb==0.23.0" \
            "tensorboard==2.20.0" "dm-tree==0.1.9" "lz4==4.4.5" \
            "pyarrow==22.0.0" "msgpack==1.1.2" "tensorboardX==2.6.4" \
            "sb3_contrib==2.7.0" "stable_baselines3==2.7.0" "einops==0.8.2"

# 4b) CRITICAL: patch ray's split_and_zero_pad to handle Dict observation spaces.
# ROOT CAUSE of the "T=2128 not chunked" learner crash (ValueError: not broadcastable
# torch.Size([B,128]) vs torch.Size([B,2128])): stock ray 2.40.0's split_and_zero_pad has
# NO nested-Dict branch, so a Dict obs is treated as ONE unsplittable item (obs stays full
# episode length while actions chunk to max_seq_len). The workstation .venv shipped a PATCHED
# zero_padding.py with _is_batched_struct/_slice_struct helpers that DO split Dict-obs leaves;
# a fresh pip install does not. (It is NOT pyarrow/msgpack — that earlier pin did not fix it.)
RAY_DIR=$(python -c "import ray,os;print(os.path.dirname(ray.__file__))")
cp "$REPO/isambard/patches/ray_zero_padding_dictobs.py" \
   "$RAY_DIR/rllib/utils/postprocessing/zero_padding.py"
python -c "import ray,os; f=os.path.join(os.path.dirname(ray.__file__),'rllib/utils/postprocessing/zero_padding.py'); assert '_is_batched_struct' in open(f).read(), 'zero_padding patch FAILED'; print('[setup] ray zero_padding Dict-obs patch applied OK')"

# 5) Build the Java gateway jar (CloudSim bytecode is arch-independent; gradle fetches an aarch64 dist)
cd "$REPO/cloudsimplus-gateway"
./gradlew compileJava jar -x test

# 6) sanity (CPU-only here; verify CUDA on a GPU node, see below)
python - <<'PY'
import torch, ray
from ray.rllib.algorithms.ppo import PPO
print("torch", torch.__version__, "| ray", ray.__version__, "| rllib import OK")
PY
echo
echo "SETUP OK. Now verify CUDA is visible on a GPU compute node:"
echo "  srun --nodes=1 --gpus=1 --time=00:10:00 --pty bash -lc \\"
echo "    'source ~/miniforge3/bin/activate rl; python -c \"import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))\"'"
