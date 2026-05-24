#!/bin/bash
# ============================================================
# One-time setup script — run interactively on a login node:
#   bash slurm/setup_env.sh
# ============================================================
set -e

SCRATCH="/scratch/gilbreth/$USER"
ENV_NAME="feedback-atlas"
PROJECT_DIR="$HOME/StudentFeedbackMinority"

echo "=== Feedback Atlas — Environment Setup ==="

# ── Load anaconda ──────────────────────────────────────────
module load anaconda

# ── Create conda environment ───────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[skip] Conda env '${ENV_NAME}' already exists."
else
    echo "[create] Creating conda env '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

source activate "${ENV_NAME}"

# ── Install PyTorch with CUDA 12.x ─────────────────────────
echo "[install] PyTorch + CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Install training requirements ─────────────────────────
echo "[install] Training requirements..."
pip install -r "${PROJECT_DIR}/training/requirements_train.txt"

# ── Download spaCy NER model ───────────────────────────────
echo "[install] spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm

# ── Pre-download HuggingFace models to scratch ────────────
# This avoids repeated downloads inside SLURM jobs and respects
# home-directory quota.
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH}/hf_cache/datasets"
mkdir -p "${HF_HOME}"

echo "[download] distilroberta-base..."
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
           AutoTokenizer.from_pretrained('distilroberta-base'); \
           AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=3)"

echo "[download] all-MiniLM-L6-v2 (sentence embeddings)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "[download] facebook/bart-large-mnli (zero-shot labeling)..."
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

echo ""
echo "==================================================="
echo " Setup complete!"
echo " Conda env  : ${ENV_NAME}"
echo " HF cache   : ${HF_HOME}"
echo " Next steps :"
echo "   sbatch slurm/train_catme.sh"
echo "   sbatch slurm/train_student.sh"
echo "==================================================="
