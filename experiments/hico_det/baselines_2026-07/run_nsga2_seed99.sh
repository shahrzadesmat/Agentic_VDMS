#!/bin/bash
#SBATCH --job-name=hico_nsga2_s99
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/Agentic_VDMS/eci/logs/hico_nsga2_s99_%j.log
#SBATCH --error=/path/to/Agentic_VDMS/eci/logs/hico_nsga2_s99_%j.err
#
# NSGA-II — evolutionary baseline (Deb et al., 2002), single-objective on SIEVE.
# population_size=10 gives ~5 generations over 50 iterations. Optuna defaults to
# 50, which at this budget is ONE generation, i.e. random search. Do not omit.
#
# Tests whether crossover can recombine into the coupled (k=50, alpha=0.80)
# optimum without modelling the coupling explicitly.
#
# Protocol matches run_gpbo_seed42.sh (50 iters, patience 0, vitl14); note the
# GP-BO runs used --exclude=gpub066,gpub088, which is no longer set.
#
# Usage: sbatch run_nsga2_seed42.sh

set -u

SEED=99
PORT=55721

INSTANCE="vdms_nsga2_s${SEED}_${SLURM_JOB_ID}"
DATASET_DIR="/path/to/datasets"
CONTAINER="/path/to/vdms_latest.sif"
PYTHON="/path/to/venv/bin/python"
DB_ROOT="/tmp/vdms_nsga2_s${SEED}_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_nsga2_s${SEED}_${SLURM_JOB_ID}.json"
FINAL_RESULTS="/path/to/Agentic_VDMS/semantic_vdms/FINAL_RESULTS/hico_det/results"
OUTPUT="${FINAL_RESULTS}/nsga2_seed${SEED}.json"

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
        > /tmp/vdms_nsga2_s${SEED}_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port $PORT"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID SEED

start_vdms

PYTHONUNBUFFERED=1 \
    $PYTHON hico_agent_optimizer_nsga2.py \
        --port "$PORT" \
        --dataset-dir "$DATASET_DIR" \
        --method nsga2 \
        --nsga2-population 10 \
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
