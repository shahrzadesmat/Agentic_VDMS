#!/bin/bash
#SBATCH --job-name=hico_llm3_llama
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hico_det/hico_llm3_s42_%j.log
#SBATCH --error=hico_det/hico_llm3_s42_%j.err

# LLM backbone ablation: Llama-3.3-70B (meta-llama/llama-3.3-70b-instruct), seed=42, 50 iter
# Post-SIEVE + all bug fixes codebase.

export OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY

PORT=55651
INSTANCE="vdms_llm3_s42_${SLURM_JOB_ID}"
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
DB_ROOT="/tmp/vdms_llm3_s42_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_llm3_s42_${SLURM_JOB_ID}.json"
FINAL_RESULTS="${BASE_DIR}/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/llm3_seed42.json"

echo "===== LLM3 Llama-3.3-70B seed=42 (post-SIEVE + all bug fixes) ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "$FINAL_RESULTS"
cd "$BASE_DIR/hico_det"

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
    apptainer instance start --no-init --nv \
        --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" \
        > /tmp/vdms_llm3_s42_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port $PORT"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID

start_vdms

PYTHONUNBUFFERED=1 \
    $PYTHON hico_agent_optimizer.py \
        --port "$PORT" \
        --dataset-dir "$DATASET_DIR" \
        --method hyperparameter_only \
        --iterations 50 \
        --seed 42 \
        --output "$OUTPUT" \
        --model "meta-llama/llama-3.3-70b-instruct" \
        --patience 0 \
        --clip-backbone vitl14 \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
