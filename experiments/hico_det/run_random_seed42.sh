#!/bin/bash
#SBATCH --job-name=hico_rand_r_s42
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:59:00
#SBATCH --exclude=gpub066,gpub088
#SBATCH --output=hico_det/hico_rand_r_s42_%j.log
#SBATCH --error=hico_det/hico_rand_r_s42_%j.err

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"

PORT=55637
INSTANCE="vdms_rand_r_s42_${SLURM_JOB_ID}"
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
DB_ROOT="/tmp/vdms_rand_r_s42_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_rand_r_s42_${SLURM_JOB_ID}.json"
FINAL_RESULTS="/work/hdd/bdjd/vdms_workflow/semantic_vdms/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/random_seed42.json"

echo "===== Random search rerun seed=42 ====="
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
        > /tmp/vdms_rand_r_s42_${SLURM_JOB_ID}.log 2>&1 &
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
        --method random \
        --iterations 50 \
        --seed 42 \
        --output "$OUTPUT" \
        --model "minimax/minimax-m2.1" \
        --patience 0 \
        --clip-backbone vitl14 \
        \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
