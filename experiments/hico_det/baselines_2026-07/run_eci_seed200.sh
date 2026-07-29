#!/bin/bash
#SBATCH --job-name=hico_eci_s200
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/Agentic_VDMS/eci/logs/hico_eci_s200_%j.log
#SBATCH --error=/path/to/Agentic_VDMS/eci/logs/hico_eci_s200_%j.err
#
# ECI — constrained Bayesian optimization baseline (Gardner et al., ICML 2014).
# Objective GP models raw QPS; feasibility (mAP >= tau) is a separate constraint
# GP, composed by Optuna as ConstrainedLogEI.
#
# Isolates cliff-modelling from cross-stage (k, alpha) coupling: ECI removes the
# cliff handicap that GP-BO suffers while still having no joint reasoning.
#
# Protocol matches run_gpbo_seed42.sh (50 iters, patience 0, vitl14); note the
# GP-BO runs used --exclude=gpub066,gpub088, which is no longer set.
#
# Usage: sbatch run_eci_seed200.sh

set -u

SEED=200
PORT=55712

INSTANCE="vdms_eci_s${SEED}_${SLURM_JOB_ID}"
DATASET_DIR="/path/to/datasets"
CONTAINER="/path/to/vdms_latest.sif"
PYTHON="/path/to/venv/bin/python"
DB_ROOT="/tmp/vdms_eci_s${SEED}_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_eci_s${SEED}_${SLURM_JOB_ID}.json"
FINAL_RESULTS="/path/to/Agentic_VDMS/semantic_vdms/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/eci_seed${SEED}.json"

echo "===== ECI (constrained BO) seed=${SEED} ====="
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
        > /tmp/vdms_eci_s${SEED}_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port $PORT"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID SEED

start_vdms

PYTHONUNBUFFERED=1 \
    $PYTHON hico_agent_optimizer_eci.py \
        --port "$PORT" \
        --dataset-dir "$DATASET_DIR" \
        --method eci \
        --iterations 50 \
        --seed "$SEED" \
        --output "$OUTPUT" \
        --patience 0 \
        --clip-backbone vitl14 \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
