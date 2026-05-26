#!/bin/bash
# ============================================================
# Train Feedback Atlas Student→Professor classifier.
#
# Training data : courseEval.csv  (RMP, ~150k sampled rows)
# Validation    : studentdataset.csv  (Purdue courseeval — held-out)
#
# Two-stage:
#   Stage 1 — zero-shot label refinement via facebook/bart-large-mnli
#             (seeds from RMP emotional_label first → fewer API calls)
#             Labels are cached → interrupted jobs resume cleanly.
#   Stage 2 — Fine-tune distilroberta-base (24-class professor labels).
#
# Estimated wall time: ~8-12 h (150k rows; GPU-accelerated zero-shot)
# Submit:  sbatch slurm/train_professor.sh
# ============================================================

#SBATCH --job-name=feedback-professor
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gilbreth/%u/slurm-feedback-professor-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aelmersa@purdue.edu

set -e

# ── Paths ─────────────────────────────────────────────────
SCRATCH="/scratch/gilbreth/$USER"
PROJECT_DIR="$HOME/StudentFeedbackMinority"
OUTPUT_DIR="${PROJECT_DIR}/professor_feedback_classifier"
ZS_CACHE="${SCRATCH}/professor_zs_labels_cache.json"
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
conda activate feedback-atlas

# Point HuggingFace to existing 39G cache in scratch
export HF_HOME="${SCRATCH}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false

# ── Verify GPU ────────────────────────────────────────────
echo "[info] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python -c "import torch; print(f'[info] CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── Run training ──────────────────────────────────────────
cd "${PROJECT_DIR}"

python -m training.train \
    --mode student_to_professor \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/professor_train_${SLURM_JOB_ID}.log"

# ── Report ────────────────────────────────────────────────
echo "============================================================"
echo " Training complete: $(date)"
echo " Model saved to  : ${OUTPUT_DIR}"
echo " Courseeval validation accuracy printed above"
echo "============================================================"
