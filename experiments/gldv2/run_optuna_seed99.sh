#!/bin/bash
#SBATCH --job-name=gldv2_opt_s99
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# GLDv2 OPTUNA — 50 iterations, seed=99

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit the four variables below to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"  # root of semantic_vdms/
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"              # dataset root
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

SEED=99
PORT=55683
INSTANCE="vdms_gldv2_opt_s99_${SLURM_JOB_ID}"
DB_ROOT="/tmp/vdms_gldv2_opt_s99_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_gldv2_opt_s99_${SLURM_JOB_ID}.json"
OUTPUT="${BASE_DIR}/supplementary/results/gldv2/optuna/seed99.json"

echo "===== GLDv2 optuna seed=99 ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "$(dirname "$OUTPUT")"
cd "/work/hdd/bdjd/vdms_workflow/src/gldv2"

for f in gldv2_clip.npy gldv2_dinov2.npy gldv2_ids.json \
          gldv2_query_clip.npy gldv2_query_dinov2.npy gldv2_query_ids.json \
          gldv2_gt.json; do
    if [ ! -f "${DATASET_DIR}/gldv2/${f}" ]; then
        echo "ERROR: Missing ${DATASET_DIR}/gldv2/${f}"; exit 1
    fi
done
echo "Prerequisites verified."

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
    apptainer instance start --no-init --nv --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    if [ $? -ne 0 ]; then echo "ERROR: Failed to start Apptainer instance"; exit 1; fi
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" > /tmp/vdms_gldv2_opt_s99_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: VDMS did not become ready within 180 seconds"
        cat /tmp/vdms_gldv2_opt_s99_${SLURM_JOB_ID}.log
        apptainer instance stop "$INSTANCE" 2>/dev/null; exit 1
    fi
    echo "[VDMS] Ready on port $PORT (fresh DB)"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID SEED

start_vdms

PYTHONUNBUFFERED=1 $PYTHON gldv2_agent_optimizer.py \
    --port "$PORT" \
    --dataset-dir "$DATASET_DIR" \
    --method optuna \
    --iterations 50 \
    --seed "$SEED" \
    --output "$OUTPUT" \
    --patience 0 \
    --vdms-restart-cmd "bash -c start_vdms"
STATUS=$?

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
