#!/bin/bash
#SBATCH --job-name=mil_sift_gpbo_s99
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit these variables to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
PYTHON=/path/to/venv/bin/python         # Python interpreter
SRC_ROOT=/path/to/Agentic_VDMS/src                # repo src/ directory
DATASET=/path/to/datasets                     # dataset root
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1
export $(grep -E '^export OPENROUTER_API_KEY=' ~/.bashrc | tail -1) 2>/dev/null || true
export LLM_MODEL="minimax/minimax-m2.1"

echo "=== Milvus SIFT1M gpbo seed=99 ===" && date

/path/to/venv/bin/python $SRC_ROOT/milvus/milvus_sift1m_optimizer.py \
    --dataset-dir $DATASET \
    --method      gp_bo \
    --iterations  50 \
    --seed        99 \
    --recall-threshold 0.90 \
    --output      /path/to/Agentic_VDMS/milvus/sift1m/results/milvus_gpbo_s99.json

echo "=== Done ===" && date
