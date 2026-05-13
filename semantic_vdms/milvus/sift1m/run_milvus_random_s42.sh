#!/bin/bash
#SBATCH --job-name=mil_sift_random_s42
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpub066,gpub088
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/sift1m/logs/sift1m_random_s42_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/sift1m/logs/sift1m_random_s42_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export $(grep -E '^export OPENROUTER_API_KEY=' ~/.bashrc | tail -1) 2>/dev/null || true
export LLM_MODEL="minimax/minimax-m2.1"

echo "=== Milvus SIFT1M random seed=42 ===" && date

/work/hdd/bdjd/vdms_code/venv/bin/python /work/hdd/bdjd/vdms_workflow/src/milvus/milvus_sift1m_optimizer.py \
    --dataset-dir /work/hdd/bdjd/vdms/datasets \
    --method      random \
    --iterations  50 \
    --seed        42 \
    --recall-threshold 0.90 \
    --output      /work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/sift1m/results/milvus_random_s42.json

echo "=== Done ===" && date
