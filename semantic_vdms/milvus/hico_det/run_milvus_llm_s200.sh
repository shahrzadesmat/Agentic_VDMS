#!/bin/bash
#SBATCH --job-name=milvus_hico_llm_s200
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_llm_s200_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_llm_s200_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export $(grep -E '^export OPENROUTER_API_KEY=' ~/.bashrc | tail -1) 2>/dev/null || true

PYTHON=/work/hdd/bdjd/vdms_code/venv/bin/python
SRC=/work/hdd/bdjd/vdms_workflow/src/milvus
RESULTS=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/results
DATASET=/work/hdd/bdjd/vdms/datasets

echo "=== Milvus HICO-DET LLM Optimizer (seed=200) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
date

$PYTHON $SRC/milvus_hico_optimizer.py \
    --dataset-dir  $DATASET \
    --method       hyperparameter_only \
    --iterations   50 \
    --seed         200 \
    --map-threshold 0.15 \
    --output       $RESULTS/milvus_llm_s200.json

echo "=== Done ===" && date
