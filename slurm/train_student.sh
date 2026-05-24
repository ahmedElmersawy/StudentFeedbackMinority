#!/bin/bash
# ============================================================
# Train Feedback Atlas classifier on original studentdataset.csv
# (rating-based label derivation — no zero-shot needed).
#
# Estimated wall time: ~1-2 h
# Submit:  sbatch slurm/train_student.sh
# ============================================================

#SBATCH --job-name=feedback-student
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/%u/slurm-feedback-student-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aelmersa@purdue.edu

set -e

# ── Paths ─────────────────────────────────────────────────
SCRATCH="/scratch/gilbreth/$USER"
PROJECT_DIR="$HOME/StudentFeedbackMinority"
OUTPUT_DIR="${PROJECT_DIR}/final_feedback_classifier"
LOG_DIR="${SCRATCH}/feedback_logs"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Start    : $(date)"
echo "============================================================"

# ── Environment ───────────────────────────────────────────
module load anaconda
source activate feedback-atlas

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export TOKENIZERS_PARALLELISM=false

# ── Verify GPU ────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'[info] CUDA: {torch.cuda.is_available()}')"

# ── Run training ──────────────────────────────────────────
cd "${PROJECT_DIR}"

python -m training.train \
    --csv studentdataset.csv \
    --output-dir "${OUTPUT_DIR}" \
    --no-zero-shot \
    2>&1 | tee "${LOG_DIR}/student_train_${SLURM_JOB_ID}.log"

echo "============================================================"
echo " Done: $(date)"
echo " Model: ${OUTPUT_DIR}"
echo "============================================================"
