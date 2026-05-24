#!/bin/bash
# ============================================================
# Train Feedback Atlas classifier on CATME peer-review data.
#
# Two-stage:
#   Stage 1 — zero-shot label generation (facebook/bart-large-mnli)
#             Labels are cached → interrupted jobs resume cleanly.
#   Stage 2 — Fine-tune distilroberta-base with class-weighted loss.
#
# Estimated wall time: ~6-8 h (144K rows; GPU-accelerated zero-shot)
# Submit:  sbatch slurm/train_catme.sh
# ============================================================

#SBATCH --job-name=feedback-catme
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gilbreth/%u/slurm-feedback-catme-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aelmersa@purdue.edu

set -e

# ── Paths ─────────────────────────────────────────────────
SCRATCH="/scratch/gilbreth/$USER"
PROJECT_DIR="$HOME/StudentFeedbackMinority"
OUTPUT_DIR="${PROJECT_DIR}/catme_feedback_classifier"
ZS_CACHE="${SCRATCH}/catme_zs_labels_cache.json"
LOG_DIR="${SCRATCH}/feedback_logs"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " Project     : ${PROJECT_DIR}"
echo " Output dir  : ${OUTPUT_DIR}"
echo " ZS cache    : ${ZS_CACHE}"
echo " Start       : $(date)"
echo "============================================================"

# ── Environment ───────────────────────────────────────────
module load anaconda
source activate feedback-atlas

# Point HuggingFace to scratch so home quota isn't hit
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH}/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

# ── Verify GPU ────────────────────────────────────────────
echo "[info] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python -c "import torch; print(f'[info] CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── Run training ──────────────────────────────────────────
cd "${PROJECT_DIR}"

python -m training.train \
    --csv CATMEcomments_Training.csv \
    --output-dir "${OUTPUT_DIR}" \
    --zero-shot-cache "${ZS_CACHE}" \
    2>&1 | tee "${LOG_DIR}/catme_train_${SLURM_JOB_ID}.log"

echo "============================================================"
echo " Training complete: $(date)"
echo " Model saved to: ${OUTPUT_DIR}"
echo "============================================================"
