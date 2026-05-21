#!/bin/bash
#SBATCH --job-name=sift1m_llm_s99
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# SIFT1M LLM — 50 iterations, seed=99

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit the four variables below to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
BASE_DIR="/path/to/Agentic_VDMS"  # root of semantic_vdms/
DATASET_DIR="/path/to/datasets"              # dataset root
CONTAINER="/path/to/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/path/to/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_ROOT="/path/to/Agentic_VDMS/results"          # canonical results dir (repo root)

SEED=99
PORT=55690
INSTANCE="vdms_sift1m_llm_s99_${SLURM_JOB_ID}"
DB_ROOT="/tmp/vdms_sift1m_llm_s99_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_sift1m_llm_s99_${SLURM_JOB_ID}.json"
SNAP_KEY_FILE="/tmp/vdms_snap_key_sift1m_llm_s99_${SLURM_JOB_ID}.txt"
VDMS_PARAMS_FILE="/tmp/vdms_params_sift1m_llm_s99_${SLURM_JOB_ID}.json"
OUTPUT="${RESULTS_ROOT}/sift1m/llm/seed99.json"
export SNAP_KEY_FILE VDMS_PARAMS_FILE

echo "===== SIFT1M llm seed=99 ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "$(dirname "$OUTPUT")"
cd "/path/to/Agentic_VDMS/src/sift1m"

for f in sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs; do
    if [ ! -f "${DATASET_DIR}/${f}" ]; then
        echo "ERROR: Missing ${DATASET_DIR}/${f}"; exit 1
    fi
done
echo "Prerequisites verified."

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms_snapshot() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"

    if [ -f "$VDMS_PARAMS_FILE" ]; then
        VDMS_PORT="$PORT" PARAMS_FILE="$VDMS_PARAMS_FILE" CFG_OUT="$VDMS_CFG" \
        "$PYTHON" -c "
import json, os
p = json.load(open(os.environ['PARAMS_FILE']))
params = p.get('params', {})
cfg = {'port': int(os.environ['VDMS_PORT']), 'db_root_path': '/db', 'max_simultaneous_clients': 100}
if 'M' in params:              cfg['hnsw_M']             = params['M']
if 'efConstruction' in params: cfg['hnsw_efConstruction'] = params['efConstruction']
if 'efSearch' in params:       cfg['hnsw_efsearch']       = params['efSearch']
with open(os.environ['CFG_OUT'], 'w') as f: json.dump(cfg, f, indent=2)
"
        echo "[VDMS config] $(cat $VDMS_CFG)"
    fi

    apptainer instance start --no-init --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    if [ $? -ne 0 ]; then echo "ERROR: Failed to start VDMS"; exit 1; fi
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" > /tmp/vdms_sift1m_llm_s99_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: VDMS did not become ready within 180 seconds"
        apptainer instance stop "$INSTANCE" 2>/dev/null; exit 1
    fi
    echo "[VDMS] Ready on port $PORT (fresh DB)"
}
export -f start_vdms_snapshot
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID SNAP_KEY_FILE VDMS_PARAMS_FILE PYTHON

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
apptainer instance start --no-init --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
if [ $? -ne 0 ]; then echo "ERROR: Failed to start Apptainer instance"; exit 1; fi
apptainer exec instance://${INSTANCE} \
    /vdms/build/vdms -cfg "$VDMS_CFG" > /tmp/vdms_sift1m_llm_s99_${SLURM_JOB_ID}.log 2>&1 &
timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: VDMS did not become ready within 180 seconds"
    apptainer instance stop "$INSTANCE" 2>/dev/null; exit 1
fi
echo "[VDMS] Initial start ready on port $PORT"

PYTHONUNBUFFERED=1 $PYTHON sift1m_agent_optimizer.py \
    --port "$PORT" \
    --dataset-dir "$DATASET_DIR" \
    --method hyperparameter_only \
    --iterations 50 \
    --seed "$SEED" \
    --patience 0 \
    --model "minimax/minimax-m2.1" \
    --recall-threshold 0.90 \
    --vdms-restart-cmd "bash -c start_vdms_snapshot" \
    --output "$OUTPUT"
STATUS=$?

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
