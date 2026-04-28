#!/bin/bash
#SBATCH --job-name=hico_ablA_s42
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/hico_det/hico_ablation_no_history_%a_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/hico_det/hico_ablation_no_history_%a_%j.err

# ABLATION A: No History Conditioning (RERUN — batch fix + 50 iters + SIEVE metric)
# --ablation-no-history: LLM sees iteration number and phase but NO past results.
# Tests whether history conditioning is the mechanism that finds k=50/alpha=0.80/constraint=object.
# Compare against full LLM agent (mean Score=300.29 across seeds 42,99,200).

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_KEY_HERE}"


SEED=42
PORT=55670

INSTANCE="vdms_hico_ablA_s${SEED}"
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
DB_ROOT="/tmp/vdms_hico_ablA_s${SEED}_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_hico_ablA_s${SEED}_${SLURM_JOB_ID}.json"
FINAL_RESULTS="${BASE_DIR}/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/ablation_no_history_seed${SEED}.json"

echo "========================================="
echo "HICO-DET Ablation A: No History Conditioning (RERUN)"
echo "  Seed: $SEED  Port: $PORT"
echo "  --ablation-no-history: LLM cannot learn from past results"
echo "  Baseline: full LLM mean Score=300.29 (50 iters, batch mode, SIEVE metric)"
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="

mkdir -p "$FINAL_RESULTS"
cd "$BASE_DIR/hico_det"

for f in hico_clipvitl14_db.npy hico_clipvitl14_text_queries.npy \
          hico_clip_text_key_order.json cpr_results.json \
          hico_dinov2.npy hico_hoi_index.json hico_gt.json hico_sift.npy; do
    if [ ! -f "${DATASET_DIR}/hico_det/${f}" ] && [ ! -f "${BASE_DIR}/hico_det/${f}" ]; then
        echo "ERROR: Missing ${f}"; exit 1
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
        /vdms/build/vdms -cfg "$VDMS_CFG" > /tmp/vdms_hico_ablA_s${SEED}_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: VDMS did not become ready within 180 seconds"
        cat /tmp/vdms_hico_ablA_s${SEED}_${SLURM_JOB_ID}.log
        apptainer instance stop "$INSTANCE" 2>/dev/null; exit 1
    fi
    echo "[VDMS] Ready on port $PORT (fresh DB)"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID SEED

start_vdms

PYTHONUNBUFFERED=1 $PYTHON hico_agent_optimizer.py \
    --port "$PORT" \
    --dataset-dir "$DATASET_DIR" \
    --method hyperparameter_only \
    --iterations 50 \
    --seed "$SEED" \
    --output "$OUTPUT" \
    --model "minimax/minimax-m2.1" \
    --clip-backbone vitl14 \
    --patience 0 \
    --ablation-no-history \
    --vdms-restart-cmd "bash -c start_vdms"
STATUS=$?

echo "Finished: $(date)  Exit: $STATUS"
echo "Results: $OUTPUT"

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"

exit $STATUS
