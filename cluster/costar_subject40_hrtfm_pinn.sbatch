#!/bin/bash
#SBATCH --job-name=hrtfm-pinn-s40
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu_h200
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm/%x.%N.%j.out
#SBATCH --error=runs/slurm/%x.%N.%j.err

# Conda activation hooks may inspect unset variables, so enable nounset only
# after the environment has been activated.
set -eo pipefail

# Submit this file from the hrtfm_pinn repository root.  These variables
# can be overridden at submission time, for example:
# HUTUBS_MAT=/users/yc01847/datasets/40.mat sbatch cluster/...
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
HUTUBS_MAT="${HUTUBS_MAT:-${PROJECT_DIR}/../PINN-for-HRTF-upsampling/40.mat}"
MA_PINN_MAT="${MA_PINN_MAT:-${PROJECT_DIR}/../PINN-for-HRTF-upsampling/40_L3.mat}"
CONDA_ROOT="${CONDA_ROOT:-/users/${USER}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hrtfm_pinn}"

SEED=2026
SAMPLING_SEED=2040
INTEGRATION_STEPS=40
NUM_SAMPLES=5
RUN_NAME="hrtfm_pinn_seed${SEED}_job_${SLURM_JOB_ID}"
RUN_ROOT="${PROJECT_DIR}/runs/hutubs_subject40/${RUN_NAME}"
TRAIN_DIR="${RUN_ROOT}/training"
EVAL_DIR="${RUN_ROOT}/evaluation/heun${INTEGRATION_STEPS}_samples${NUM_SAMPLES}"
BASELINE_DIR="${RUN_ROOT}/baselines/fei_ma_pinn"

if [[ ! -f "${PROJECT_DIR}/scripts/train_hrtf.py" ]]; then
    echo "Submit from the hrtfm_pinn repository root; got ${PROJECT_DIR}" >&2
    exit 2
fi
if [[ ! -f "${HUTUBS_MAT}" ]]; then
    echo "HUTUBS subject-40 file not found: ${HUTUBS_MAT}" >&2
    exit 2
fi
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found under ${CONDA_ROOT}" >&2
    exit 2
fi

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${PROJECT_DIR}"

{
    echo "run_name=${RUN_NAME}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "slurm_job_name=${SLURM_JOB_NAME}"
    echo "slurm_node_list=${SLURM_JOB_NODELIST}"
    echo "project_dir=${PROJECT_DIR}"
    echo "hutubs_mat=${HUTUBS_MAT}"
    echo "conda_environment=${CONDA_ENV_NAME}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} > "${RUN_ROOT}/run_manifest.txt"

conda list --explicit > "${RUN_ROOT}/conda_explicit.txt"
nvidia-smi > "${RUN_ROOT}/gpu_info.txt"
git status --short > "${RUN_ROOT}/git_status.txt" 2>/dev/null || true

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated a GPU, but PyTorch cannot access CUDA")
print(f"PyTorch {torch.__version__}")
print(f"CUDA runtime {torch.version.cuda}")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY

python -m unittest discover -s tests -v
python tools/py_syntax_check.py

TRAIN_COMMAND=(
    python -u scripts/train_hrtf.py
    --input "${HUTUBS_MAT}"
    --output-dir "${TRAIN_DIR}"
    --steps 20000
    --batch-size 4
    --learning-rate 2e-4
    --weight-decay 1e-4
    --grad-clip 1.0
    --seed "${SEED}"
    --device cuda:0
    --num-workers 0
    --log-every 100
    --save-every 1000
    --width 128
    --depth 4
    --context-dim 128
    --observation-width 96
    --coordinate-bands 4
    --time-dim 32
    --frequency-scale-hz 15000
    --prior-modes 16
    --prior-seed 1729
    --physics-weight 0.1
    --physics-warmup-steps 500
    --physics-ramp-steps 2000
    --physics-every 1
    --physics-points 64
    --physics-unroll-steps 4
    --speed-of-sound 343
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${RUN_ROOT}/training_command.txt"
printf '\n' >> "${RUN_ROOT}/training_command.txt"
"${TRAIN_COMMAND[@]}"

EVAL_COMMAND=(
    python -u scripts/sample_hrtf.py
    --input "${HUTUBS_MAT}"
    --checkpoint "${TRAIN_DIR}/checkpoint_final.pt"
    --output-dir "${EVAL_DIR}"
    --num-samples "${NUM_SAMPLES}"
    --integration-steps "${INTEGRATION_STEPS}"
    --seed "${SAMPLING_SEED}"
    --device cuda:0
)
printf '%q ' "${EVAL_COMMAND[@]}" > "${RUN_ROOT}/evaluation_command.txt"
printf '\n' >> "${RUN_ROOT}/evaluation_command.txt"
"${EVAL_COMMAND[@]}"

# If Fei Ma's reference prediction is beside 40.mat, evaluate it with the same
# implementation for an immediately comparable baseline.  Its absence does not
# invalidate the PI-FM run.
if [[ -f "${MA_PINN_MAT}" ]]; then
    mkdir -p "${BASELINE_DIR}"
    python -u scripts/evaluate_hrtf.py \
        --reference "${HUTUBS_MAT}" \
        --prediction "${MA_PINN_MAT}" \
        --prediction-key total_est \
        --output-dir "${BASELINE_DIR}"
else
    echo "Optional Fei Ma baseline not found; skipped: ${MA_PINN_MAT}"
fi

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_ROOT}/run_manifest.txt"
echo "Completed run: ${RUN_ROOT}"
