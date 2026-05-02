#!/bin/bash
#SBATCH --job-name=milvus_hico_llm_s99
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_llm_s99_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_llm_s99_%j.err
#SBATCH --exclude=gpub066,gpub088

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/work/hdd/bdjd/vdms_code/venv/bin/python
SRC=/work/hdd/bdjd/vdms_workflow/src/milvus
RESULTS=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/results
DATASET=/work/hdd/bdjd/vdms/datasets

[ -f /work/hdd/bdjd/vdms_workflow/.env ] && source /work/hdd/bdjd/vdms_workflow/.env

echo "=== Milvus HICO-DET LLM Optimizer (seed=99) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
date

$PYTHON $SRC/milvus_hico_optimizer.py \
    --dataset-dir  $DATASET \
    --method       hyperparameter_only \
    --iterations   50 \
    --seed         99 \
    --map-threshold 0.15 \
    --output       $RESULTS/milvus_llm_s99.json

echo "=== Done ===" && date
