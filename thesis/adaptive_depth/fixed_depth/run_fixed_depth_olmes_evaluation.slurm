#!/bin/bash
#SBATCH --job-name=bolmo_fd_eval
#SBATCH --output=logs/adaptive_depth/fixed_depth/fixed_depth_eval_%j.out
#SBATCH --error=logs/adaptive_depth/fixed_depth/fixed_depth_eval_%j.err
#SBATCH --account=lp_dtai1
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "===== Fixed-depth OLMES evaluation started on $(date) ====="
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

echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
nvcc --version || true
nvidia-smi || true

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT="/vsc-hard-mounts/leuven-data/382/vsc38215/master_thesis"

BOLMO_CORE="${ROOT}/bolmo-core"
OLMES_DIR="${ROOT}/olmes"
VENV="${BOLMO_CORE}/.venv"

MODEL_ID="${VSC_SCRATCH}/master_thesis/checkpoints/Bolmo-1B"

SOURCE_MODELING_FILE="${BOLMO_CORE}/src/olmo_core/nn/bolmo/hf/modeling_bolmo.py"
TARGET_MODELING_FILE="${MODEL_ID}/modeling_bolmo.py"

OLMES_BOLMO_FILE="${OLMES_DIR}/oe_eval/models/eleuther_huggingface_bolmo.py"
OLMES_DEFAULTS_FILE="${OLMES_DIR}/oe_eval/default_configs.py"

EVAL_SCRIPT="${BOLMO_CORE}/thesis/adaptive_depth/fixed_depth/evaluate_fixed_depth_olmes.py"

OUT_DIR="${ROOT}/outputs/adaptive_depth/fixed_depth/evaluation_${SLURM_JOB_ID}"
CACHE_DIR="${VSC_SCRATCH}/master_thesis/hf_cache"

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
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------
# Environment sanity checks
# ---------------------------------------------------------------------
echo ""
echo "===== Environment sanity checks ====="

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

echo "olmes executable: $(which olmes || echo 'NOT FOUND')"
olmes --help >/dev/null

echo ""
echo "ROOT=${ROOT}"
echo "BOLMO_CORE=${BOLMO_CORE}"
echo "OLMES_DIR=${OLMES_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "SOURCE_MODELING_FILE=${SOURCE_MODELING_FILE}"
echo "TARGET_MODELING_FILE=${TARGET_MODELING_FILE}"
echo "OLMES_BOLMO_FILE=${OLMES_BOLMO_FILE}"
echo "OLMES_DEFAULTS_FILE=${OLMES_DEFAULTS_FILE}"
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"

# ---------------------------------------------------------------------
# Required-file checks
# ---------------------------------------------------------------------
echo ""
echo "===== Required file checks ====="

test -d "${MODEL_ID}"
echo "Found local HF Bolmo checkpoint."

test -f "${SOURCE_MODELING_FILE}"
echo "Found modified source modeling_bolmo.py."

test -f "${OLMES_BOLMO_FILE}"
echo "Found patched OLMES Bolmo backend file."

test -f "${OLMES_DEFAULTS_FILE}"
echo "Found OLMES default config file."

test -f "${EVAL_SCRIPT}"
echo "Found fixed-depth OLMES evaluation driver."

# ---------------------------------------------------------------------
# Copy latest modified modeling_bolmo.py into local HF checkpoint
# ---------------------------------------------------------------------
echo ""
echo "===== Copying modified modeling_bolmo.py into local HF checkpoint ====="

cp "${SOURCE_MODELING_FILE}" "${TARGET_MODELING_FILE}"

echo "Copy finished:"
ls -lh "${TARGET_MODELING_FILE}"

# ---------------------------------------------------------------------
# Verify OLMES fixed-depth modifications are present
# ---------------------------------------------------------------------
echo ""
echo "===== Verifying OLMES fixed-depth integration ====="

if grep -q "fixed_depth_fraction" "${OLMES_BOLMO_FILE}"; then
    echo "OLMES Bolmo backend contains fixed_depth_fraction."
else
    echo "ERROR: fixed_depth_fraction not found in:"
    echo "${OLMES_BOLMO_FILE}"
    exit 1
fi

if grep -q '"fixed_depth_fraction"' "${OLMES_DEFAULTS_FILE}"; then
    echo "OLMES MODEL_DEFAULTS contains fixed_depth_fraction."
else
    echo "ERROR: fixed_depth_fraction not found in MODEL_DEFAULTS:"
    echo "${OLMES_DEFAULTS_FILE}"
    exit 1
fi

# ---------------------------------------------------------------------
# Verify local HF model still contains the fixed-depth hook
# ---------------------------------------------------------------------
echo ""
echo "===== Verifying local HF Bolmo fixed-depth hook ====="

python - <<'PY'
import os
from transformers import AutoModelForCausalLM

model_path = os.path.expandvars(
    "$VSC_SCRATCH/master_thesis/checkpoints/Bolmo-1B"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

print("Loaded class module:", model.__class__.__module__)
print("Has model.model._run_global_layers:", hasattr(model.model, "_run_global_layers"))
print("Has model.model.global_backbone_strategy:", hasattr(model.model, "global_backbone_strategy"))

if not hasattr(model.model, "_run_global_layers"):
    raise RuntimeError("Missing _run_global_layers in locally loaded HF Bolmo model.")

if not hasattr(model.model, "global_backbone_strategy"):
    raise RuntimeError("Missing global_backbone_strategy in locally loaded HF Bolmo model.")

print("Local HF Bolmo model hook verification passed.")
PY

# ---------------------------------------------------------------------
# Run full fixed-depth OLMES evaluation
# ---------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "Running full fixed-depth evaluation:"
echo "  Depths: 25%, 50%, 75%, 100%"
echo "  Tasks:  PIQA + HellaSwag"
echo "  Output: ${OUT_DIR}"
echo "======================================================================"

cd "${OLMES_DIR}"

python "${EVAL_SCRIPT}" \
    --model-id "${MODEL_ID}" \
    --output-dir "${OUT_DIR}" \
    --depths 0.25 0.50 0.75 1.0 \
    --tasks piqa::olmes hellaswag::olmes \
    --batch-size 8 \
    --max-length 4096 \
    --num-workers 1

echo ""
echo "===== Fixed-depth OLMES evaluation finished on $(date) ====="
echo "Results saved in: ${OUT_DIR}"