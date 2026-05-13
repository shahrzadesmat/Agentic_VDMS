#!/bin/bash
#SBATCH --job-name=mil_random_s200
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_random_s200_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_random_s200_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/work/hdd/bdjd/vdms_code/venv/bin/python
SRC=/work/hdd/bdjd/vdms_workflow/src/milvus
DATASET=/work/hdd/bdjd/vdms/datasets

echo "=== Milvus HICO-DET random seed=200 ===" && date

$PYTHON $SRC/milvus_hico_optimizer.py \
    --dataset-dir   $DATASET \
    --method        random \
    --iterations    50 \
    --seed          200 \
    --map-threshold 0.15 \
    --force-constraint none \
    --output        /work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/results/milvus_random_s200.json

echo "=== Done ===" && date
