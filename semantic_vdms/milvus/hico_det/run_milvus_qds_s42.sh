#!/bin/bash
#SBATCH --job-name=milvus_hico_qds_s42
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit these variables to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
PYTHON=/work/hdd/bdjd/vdms_code/venv/bin/python         # Python interpreter
SRC_ROOT=/work/hdd/bdjd/vdms_workflow/src                # repo src/ directory
DATASET=/work/hdd/bdjd/vdms/datasets                     # dataset root
# ─────────────────────────────────────────────────────────────────────────────
RESULTS=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/results

set -euo pipefail
export PYTHONUNBUFFERED=1
export $(grep -E '^export OPENROUTER_API_KEY=' ~/.bashrc | tail -1) 2>/dev/null || true


echo "=== Milvus HICO-DET LLM+QDS Optimizer (seed=42) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
date

$PYTHON $SRC_ROOT/milvus/milvus_hico_optimizer.py \
    --dataset-dir  $DATASET \
    --method       hyperparameter_only \
    --iterations   50 \
    --seed         42 \
    --map-threshold 0.15 \
    --use-qds-objective \
    --output       $RESULTS/milvus_qds_s42.json

echo "=== Done ===" && date
