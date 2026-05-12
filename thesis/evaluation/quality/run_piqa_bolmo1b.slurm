#!/bin/bash
#SBATCH --job-name=bolmo1b_piqa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=intro_vsc38215
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "===== Job started on $(date) ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

module purge
module load cluster/wice/gpu_a100
module load CUDA/12.6.0

export CUDA_HOME="$EBROOTCUDA"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
nvcc --version || true

# -------------------------
# Paths
# -------------------------
ROOT="/vsc-hard-mounts/leuven-data/382/vsc38215/master_thesis"
VENV="${ROOT}/bolmo-core/.venv"
OUT_DIR="${ROOT}/outputs/evaluation/quality/bolmo-1b-baseline/piqa"
CACHE_DIR="${ROOT}/outputs/hf_cache"

mkdir -p "${OUT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p logs

# -------------------------
# Environment
# -------------------------
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"

echo "Python: $(which python)"
python --version
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

echo "olmes: $(which olmes)"
olmes --help >/dev/null

# -------------------------
# Run evaluation
# -------------------------
cd "${ROOT}"
#--task hellaswag:mc::olmes \

olmes \
  --model allenai/Bolmo-1B \
  --model-args '{"max_length": 4096, "trust_remote_code": "true", "model_type": "hf_bolmo", "add_bos_token": "true"}' \
  --task piqa::olmes \
  --batch-size 8 \
  --output-dir "${OUT_DIR}"

echo "===== Job finished on $(date) ====="