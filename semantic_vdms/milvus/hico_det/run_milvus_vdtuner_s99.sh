#!/bin/bash
#SBATCH --job-name=mil_vdt_s99
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_vdtuner_s99_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/logs/milvus_vdtuner_s99_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/work/hdd/bdjd/vdms_code/venv/bin/python
SRC=/work/hdd/bdjd/vdms_workflow/src/milvus
DATASET=/work/hdd/bdjd/vdms/datasets

echo "=== Milvus HICO-DET VDTuner seed=99 ===" && date

$PYTHON $SRC/milvus_hico_optimizer.py \
    --dataset-dir   $DATASET \
    --method        vdtuner \
    --iterations    50 \
    --seed          99 \
    --map-threshold 0.15 \
    --output        /work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det/results/milvus_vdtuner_s99.json

echo "=== Done ===" && date
