#!/bin/bash
#SBATCH --job-name=bolmo1b_phase0_sanity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=lp_dtai1
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "===== Job started on $(date) ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "VSC_SCRATCH=${VSC_SCRATCH:-not_set}"

module purge
module load cluster/wice/gpu_a100_debug
module load CUDA/12.6.0

export CUDA_HOME="$EBROOTCUDA"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvidia-smi || true

# -------------------------
# Paths
# -------------------------
ROOT="/vsc-hard-mounts/leuven-data/382/vsc38215/master_thesis"
BOLMO_CORE="${ROOT}/bolmo-core"
VENV="${BOLMO_CORE}/.venv"

MODEL_ID="${VSC_SCRATCH}/master_thesis/checkpoints/Bolmo-1B"

OUT_DIR="${ROOT}/outputs/profiling/phase0_sanity_${SLURM_JOB_ID}"
CACHE_DIR="${ROOT}/outputs/hf_cache"

mkdir -p "${OUT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p logs

# -------------------------
# Check local checkpoint
# -------------------------
echo "Checking MODEL_ID=${MODEL_ID}"

if [ ! -d "${MODEL_ID}" ]; then
  echo "[error] Local checkpoint directory does not exist:"
  echo "        ${MODEL_ID}"
  echo ""
  echo "Searching for Bolmo folders in scratch:"
  find "${VSC_SCRATCH}/master_thesis" -maxdepth 5 -type d -iname "*Bolmo*" || true
  exit 1
fi

echo "Checkpoint directory exists. Contents:"
ls -lah "${MODEL_ID}" | head -50

if [ ! -f "${MODEL_ID}/config.json" ]; then
  echo "[error] ${MODEL_ID}/config.json not found."
  echo "This does not look like a valid HuggingFace checkpoint directory."
  exit 1
fi

# -------------------------
# Environment
# -------------------------
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
python --version
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import transformers; print('transformers:', transformers.__version__)"

# -------------------------
# Run Phase 0 sanity profiler
# -------------------------
cd "${BOLMO_CORE}"

echo "Running Phase 0 profiling sanity test..."
echo "MODEL_ID=${MODEL_ID}"
echo "OUT_DIR=${OUT_DIR}"

srun python thesis/profiling/profile_phase0.py \
  --model-id "${MODEL_ID}" \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --num-warmup 3 \
  --num-repeats 5 \
  --max-new-tokens 16 \
  --output-dir "${OUT_DIR}" \
  --inspect-modules

# -------------------------
# Generate plots
# -------------------------
echo "Generating plots..."

srun python thesis/profiling/analyze_phase0.py \
  --input-dir "${OUT_DIR}"

echo "===== Output directory ====="
echo "${OUT_DIR}"
ls -R "${OUT_DIR}" || true

echo "===== Job finished on $(date) ====="