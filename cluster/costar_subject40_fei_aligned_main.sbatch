#!/bin/bash
#SBATCH --job-name=hrtfm-fei-align
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu_h200
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm/%x.%N.%j.out
#SBATCH --error=runs/slurm/%x.%N.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
HUTUBS_MAT="${HUTUBS_MAT:-${PROJECT_DIR}/../PINN-for-HRTF-upsampling/40.mat}"
MA_PINN_MAT="${MA_PINN_MAT:-${PROJECT_DIR}/../PINN-for-HRTF-upsampling/results/paper-subject40-1m-20260901T111304Z/subject-40/40_L3.mat}"
CONDA_ROOT="${CONDA_ROOT:-/users/${USER}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hrtfm_pinn}"

SEED="${SEED:-2026}"
STEPS_PER_ATTEMPT="${STEPS_PER_ATTEMPT:-1000000}"
ATTEMPTS="${ATTEMPTS:-5}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
INTEGRATION_STEPS="${INTEGRATION_STEPS:-40}"
SAMPLING_SEED="${SAMPLING_SEED:-2040}"
RUN_NAME="${RUN_NAME:-hrtfm_pinn_fei_aligned_1m_seed${SEED}_job_${SLURM_JOB_ID}}"
RUN_ROOT="${PROJECT_DIR}/runs/hutubs_subject40/${RUN_NAME}"
TRAIN_DIR="${RUN_ROOT}/training"
EVAL_DIR="${RUN_ROOT}/evaluation/heun${INTEGRATION_STEPS}_samples${NUM_SAMPLES}"
BASELINE_DIR="${RUN_ROOT}/baselines/fei_ma_pinn_1m"

if [[ ! -f "${PROJECT_DIR}/scripts/train_hrtf_fei_aligned.py" ]]; then
    echo "Submit from the fei-ma-comparable branch repository root: ${PROJECT_DIR}" >&2
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
    echo "partition=main"
    echo "project_dir=${PROJECT_DIR}"
    echo "hutubs_mat=${HUTUBS_MAT}"
    echo "fei_ma_baseline=${MA_PINN_MAT}"
    echo "steps_per_attempt=${STEPS_PER_ATTEMPT}"
    echo "attempts=${ATTEMPTS}"
    echo "started_or_resumed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} >> "${RUN_ROOT}/run_manifest.txt"

conda list --explicit > "${RUN_ROOT}/conda_explicit.txt"
nvidia-smi > "${RUN_ROOT}/gpu_info.txt"
git status --short > "${RUN_ROOT}/git_status.txt" 2>/dev/null || true

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated a GPU, but PyTorch cannot access CUDA")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}")
PY
python -m unittest discover -s tests -v
python tools/py_syntax_check.py

TRAIN_COMMAND=(
    python -u scripts/train_hrtf_fei_aligned.py
    --input "${HUTUBS_MAT}"
    --output-dir "${TRAIN_DIR}"
    --steps-per-attempt "${STEPS_PER_ATTEMPT}"
    --attempts "${ATTEMPTS}"
    --depth 3
    --learning-rate 1e-3
    --physics-weight 1
    --physics-unroll-steps 4
    --prior-modes 16
    --prior-seed 1729
    --speed-of-sound 343
    --loss-threshold-db -29
    --selection-samples 5
    --selection-integration-steps 40
    --seed "${SEED}"
    --device cuda:0
    --log-every "${LOG_EVERY}"
    --save-every "${SAVE_EVERY}"
    --max-fields 28
)
if [[ -f "${TRAIN_DIR}/resume.pt" ]]; then
    TRAIN_COMMAND+=(--resume "${TRAIN_DIR}/resume.pt")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "${RUN_ROOT}/training_command.txt"
printf '\n' >> "${RUN_ROOT}/training_command.txt"
if [[ ! -f "${TRAIN_DIR}/TRAINING_COMPLETE" ]]; then
    "${TRAIN_COMMAND[@]}"
else
    echo "Training is already complete; proceeding to evaluation."
fi

EVAL_COMMAND=(
    python -u scripts/sample_hrtf_fei_aligned.py
    --input "${HUTUBS_MAT}"
    --training-dir "${TRAIN_DIR}"
    --output-dir "${EVAL_DIR}"
    --num-samples "${NUM_SAMPLES}"
    --integration-steps "${INTEGRATION_STEPS}"
    --seed "${SAMPLING_SEED}"
    --device cuda:0
)
printf '%q ' "${EVAL_COMMAND[@]}" > "${RUN_ROOT}/evaluation_command.txt"
printf '\n' >> "${RUN_ROOT}/evaluation_command.txt"
"${EVAL_COMMAND[@]}"

if [[ -f "${MA_PINN_MAT}" ]]; then
    mkdir -p "${BASELINE_DIR}"
    python -u scripts/evaluate_hrtf.py \
        --reference "${HUTUBS_MAT}" \
        --prediction "${MA_PINN_MAT}" \
        --prediction-key total_est \
        --output-dir "${BASELINE_DIR}"
else
    echo "Fei Ma 1M baseline not found; skipped: ${MA_PINN_MAT}" >&2
fi

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_ROOT}/run_manifest.txt"
echo "Completed run: ${RUN_ROOT}"
