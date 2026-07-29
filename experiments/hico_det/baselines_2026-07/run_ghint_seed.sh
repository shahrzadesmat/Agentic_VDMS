#!/bin/bash
#SBATCH --job-name=hico_ghint
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/Agentic_VDMS/eci/logs/%x_%j.log
#SBATCH --error=/path/to/Agentic_VDMS/eci/logs/%x_%j.err
#
# Component ablation: isolate the diagnostic guidance g(H) and the untried-value
# hints. Table 5 currently bundles both as "w/o phases/hints", so the paper
# claims three components matter while demonstrating two.
#
# Parameterised by environment variables so one script covers both conditions:
#   COND  = noguid | nohint
#   SEED  = 42 | 99 | 200
#   PORT  = unique per job
#
# The OpenRouter key is read from the environment and never written to disk.
# Export it in the SUBMITTING shell; --export=ALL (the sbatch default) forwards it:
#     export OPENROUTER_API_KEY=...   # then sbatch
#
# Usage (see submit_ghint.sh for the full sweep):
#   COND=noguid SEED=42 PORT=55730 sbatch --job-name=hico_ghint_noguid_s42 run_ghint_seed.sh

set -u

: "${COND:?set COND=noguid|nohint}"
: "${SEED:?set SEED=42|99|200}"
: "${PORT:?set PORT to an unused port}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY in the submitting shell before sbatch}"

case "$COND" in
    noguid) ABL_FLAG="--ablation-no-guidance" ;;
    nohint) ABL_FLAG="--ablation-no-hints" ;;
    *) echo "unknown COND=$COND"; exit 2 ;;
esac

INSTANCE="vdms_ghint_${COND}_s${SEED}_${SLURM_JOB_ID}"
DATASET_DIR="/path/to/datasets"
CONTAINER="/path/to/vdms_latest.sif"
PYTHON="/path/to/venv/bin/python"
DB_ROOT="/tmp/vdms_ghint_${COND}_s${SEED}_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_ghint_${COND}_s${SEED}_${SLURM_JOB_ID}.json"
FINAL_RESULTS="/path/to/Agentic_VDMS/semantic_vdms/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/ablation_${COND}_seed${SEED}.json"

echo "===== ablation ${COND} seed=${SEED} (${ABL_FLAG}) ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "$FINAL_RESULTS" /path/to/Agentic_VDMS/eci/logs
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
        > /tmp/vdms_ghint_${COND}_s${SEED}_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port $PORT"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID COND SEED

start_vdms

PYTHONUNBUFFERED=1 \
    $PYTHON hico_agent_optimizer_ghint.py \
        --port "$PORT" \
        --dataset-dir "$DATASET_DIR" \
        --method hyperparameter_only \
        --iterations 50 \
        --seed "$SEED" \
        --output "$OUTPUT" \
        --model "minimax/minimax-m2.1" \
        --patience 0 \
        --clip-backbone vitl14 \
        $ABL_FLAG \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
