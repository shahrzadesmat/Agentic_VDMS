#!/bin/bash
#SBATCH --job-name=hico_gpbo_r_s200
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=hico_det/hico_gpbo_r_s200_%j.log
#SBATCH --error=hico_det/hico_gpbo_r_s200_%j.err

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit the four variables below to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
BASE_DIR="/path/to/Agentic_VDMS"  # root of semantic_vdms/
DATASET_DIR="/path/to/datasets"              # dataset root
CONTAINER="/path/to/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/path/to/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

PORT=55642
INSTANCE="vdms_gpbo_r_s200_${SLURM_JOB_ID}"
DB_ROOT="/tmp/vdms_gpbo_r_s200_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_gpbo_r_s200_${SLURM_JOB_ID}.json"
FINAL_RESULTS="${BASE_DIR}/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/gpbo_seed200.json"

echo "===== GP-BO rerun seed=200 ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "$FINAL_RESULTS"
cd "/path/to/Agentic_VDMS/src/hico_det"

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
    apptainer instance start --no-init --nv \
        --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" \
        > /tmp/vdms_gpbo_r_s200_${SLURM_JOB_ID}.log 2>&1 &
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
        --method gp_bo \
        --iterations 50 \
        --seed 200 \
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
