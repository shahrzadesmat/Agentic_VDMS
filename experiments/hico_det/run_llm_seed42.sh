#!/bin/bash
#SBATCH --job-name=hico_llm_r_s42
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=hico_det/hico_llm_rerun_seed42_%j.log
#SBATCH --error=hico_det/hico_llm_rerun_seed42_%j.err

# LLM agent rerun — batch query fix (f46911a) applied.
# Old result (serial QPS~17, Score~4.34) is invalid.
# This run uses true batch FindDescriptor (all 600 in one round-trip).

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit the four variables below to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"  # root of semantic_vdms/
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"              # dataset root
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

PORT=55630
INSTANCE="vdms_llm_r_s42_${SLURM_JOB_ID}"
DB_ROOT="/tmp/vdms_llm_r_s42_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_llm_r_s42_${SLURM_JOB_ID}.json"
FINAL_RESULTS="${BASE_DIR}/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/llm_seed42.json"

echo "====================================="
echo "HICO-DET LLM Agent — RERUN (batch fix) — seed=42"
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Started: $(date)"
echo "====================================="

mkdir -p "$FINAL_RESULTS"
cd "/work/hdd/bdjd/vdms_workflow/src/hico_det"

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
    apptainer instance start --no-init --nv \
        --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" \
        > /tmp/vdms_llm_r_s42_${SLURM_JOB_ID}.log 2>&1 &
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
        --model "minimax/minimax-m2.1" \
        --patience 0 \
        --clip-backbone vitl14 \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"

echo "Finished: $(date)  Exit: $STATUS"
echo "Output: $OUTPUT"
exit $STATUS
