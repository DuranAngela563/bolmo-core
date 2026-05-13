#!/bin/bash
#SBATCH --job-name=bolmo_fixed_depth_prof
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_dtai1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:05:00
#SBATCH --output=logs/adaptive_depth/fixed_depth/fixed_depth_profiling_%j.out
#SBATCH --error=logs/adaptive_depth/fixed_depth/fixed_depth_profiling_%j.err

set -euo pipefail

echo "===== Fixed-depth profiling job started on $(date) ====="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "VSC_SCRATCH=${VSC_SCRATCH:-not_set}"

# ---------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------
module purge
module load cluster/wice/gpu_a100
module load CUDA/12.6.0

export CUDA_HOME="$EBROOTCUDA"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvidia-smi || true

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT="/vsc-hard-mounts/leuven-data/382/vsc38215/master_thesis"
BOLMO_CORE="${ROOT}/bolmo-core"
VENV="${BOLMO_CORE}/.venv"

MODEL_ID="${VSC_SCRATCH}/master_thesis/checkpoints/Bolmo-1B"

OUT_DIR="${ROOT}/outputs/adaptive_depth/fixed_depth/profiling_${SLURM_JOB_ID}"
CACHE_DIR="${VSC_SCRATCH}/master_thesis/hf_cache"

SCRIPT="${BOLMO_CORE}/thesis/adaptive_depth/fixed_depth/profile_fixed_depth.py"

# ---------------------------------------------------------------------
# Create runtime directories
# ---------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
mkdir -p "${CACHE_DIR}"

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
cd "${BOLMO_CORE}"

source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Passed into profile_fixed_depth.py as default arguments
export BOLMO_MODEL_ID="${MODEL_ID}"
export FIXED_DEPTH_OUT_DIR="${OUT_DIR}"

# ---------------------------------------------------------------------
# Environment sanity checks
# ---------------------------------------------------------------------
echo "Python: $(which python)"
python --version

python -c "
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
"

python -c "
import transformers
print('transformers:', transformers.__version__)
"

echo "ROOT=${ROOT}"
echo "BOLMO_CORE=${BOLMO_CORE}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "SCRIPT=${SCRIPT}"

# ---------------------------------------------------------------------
# Run fixed-depth profiling
# ---------------------------------------------------------------------
python "${SCRIPT}" \
    --model-id "${MODEL_ID}" \
    --local-files-only \
    --dtype bfloat16 \
    --num-warmup 3 \
    --num-repeats 5 \
    --max-new-tokens 16 \
    --prompt-set default \
    --depths 0.25 0.50 0.75 1.0 \
    --output-dir "${OUT_DIR}"

echo "===== Fixed-depth profiling job finished on $(date) ====="
echo "Results saved in: ${OUT_DIR}"