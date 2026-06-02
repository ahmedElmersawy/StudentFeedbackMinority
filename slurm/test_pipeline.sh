#!/bin/bash
# ============================================================
# Run full inference pipeline on a CSV and write output files.
#
# Usage:
#   sbatch slurm/test_pipeline.sh                          # defaults
#   sbatch slurm/test_pipeline.sh studentdataset.csv student_to_professor
#   sbatch slurm/test_pipeline.sh courseEval.csv           student_to_professor
#
# Positional args (optional):
#   $1  Input CSV  (default: studentdataset.csv)
#   $2  Mode       student_to_professor | student_to_student (default: auto)
# ============================================================

#SBATCH --job-name=feedback-test
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gilbreth/%u/slurm-feedback-test-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aelmersa@purdue.edu

set -e

# ── Args ──────────────────────────────────────────────────
INPUT_CSV="${1:-studentdataset.csv}"
FEEDBACK_MODE="${2:-}"          # empty → auto-detect

# ── Paths ─────────────────────────────────────────────────
SCRATCH="/scratch/gilbreth/$USER"
PROJECT_DIR="$HOME/StudentFeedbackMinority"
OUTPUT_DIR="${SCRATCH}/test_output_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Start    : $(date)"
echo " Input    : ${INPUT_CSV}"
echo " Mode     : ${FEEDBACK_MODE:-auto}"
echo " Output   : ${OUTPUT_DIR}"
echo "============================================================"

# ── Environment ───────────────────────────────────────────
module load anaconda
source activate feedback-atlas

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Verify GPU ────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'[info] CUDA: {torch.cuda.is_available()}')"

# ── Run pipeline ──────────────────────────────────────────
cd "${PROJECT_DIR}"

MODE_ARG=""
if [ -n "${FEEDBACK_MODE}" ]; then
    MODE_ARG="--mode ${FEEDBACK_MODE}"
fi

python - <<PYEOF
import sys, pandas as pd
from backend.pipeline import run_full_pipeline, generate_output_files, aggregate_by_entity

input_csv   = "${INPUT_CSV}"
output_dir  = "${OUTPUT_DIR}"
mode        = "${FEEDBACK_MODE}" or None

print(f"[pipeline] Running on: {input_csv}  mode={mode or 'auto'}")
df = run_full_pipeline(input_csv, feedback_mode=mode)

# ── Per-entity summary ────────────────────────────────────
summary = aggregate_by_entity(df)
print("\n=== ENTITY SUMMARY ===")
for _, row in summary.iterrows():
    attn = "NEEDS_ATTENTION" if row["needs_attention"] else "OK"
    print(f"\n[{row['entity'].upper()}]  ({attn}  {row['total_responses']} responses)")
    for f in row["flagged_findings"]:
        print(f"  - {f['label']}: {f['count']} students ({f['pct']:.1f}%)")

# ── Write output files ────────────────────────────────────
generate_output_files(df, output_dir, mode or "auto")
print(f"\n[pipeline] Output written to: {output_dir}")
PYEOF

echo "============================================================"
echo " Done: $(date)"
echo " Files:"
ls -lh "${OUTPUT_DIR}"
echo "============================================================"
