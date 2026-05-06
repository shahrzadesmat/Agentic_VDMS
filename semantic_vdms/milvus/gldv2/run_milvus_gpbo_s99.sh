#!/bin/bash
#SBATCH --job-name=mil_gldv_gpbo_s99
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpub066,gpub088
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/gldv2/logs/gldv2_gpbo_s99_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/gldv2/logs/gldv2_gpbo_s99_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export $(grep -E '^export OPENROUTER_API_KEY=' ~/.bashrc | tail -1) 2>/dev/null || true
export LLM_MODEL="minimax/minimax-m2.1"

echo "=== Milvus GLDV2 gpbo seed=99 ===" && date

/work/hdd/bdjd/vdms_code/venv/bin/python /work/hdd/bdjd/vdms_workflow/src/milvus/milvus_gldv2_optimizer.py \
    --dataset-dir /work/hdd/bdjd/vdms/datasets \
    --method      gp_bo \
    --iterations  50 \
    --seed        99 \
    --map-threshold 0.15 \
    --output      /work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/gldv2/results/milvus_gpbo_s99.json

echo "=== Done ===" && date
