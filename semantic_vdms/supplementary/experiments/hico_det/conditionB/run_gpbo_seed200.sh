#!/bin/bash
#SBATCH --job-name=hico_gpbo_cB_s200
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hico_det/conditionB/hico_gpbo_cB_s200_%j.log
#SBATCH --error=hico_det/conditionB/hico_gpbo_cB_s200_%j.err
#SBATCH --chdir=/work/hdd/bdjd/vdms_workflow/semantic_vdms/supplementary/experiments

# Condition B: multi-row VDMS indexing with all_verb_ids.

PORT=55682
INSTANCE="vdms_gpbo_cB_s200_${SLURM_JOB_ID}"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
OPTIMIZER="/work/hdd/bdjd/vdms_workflow/src/hico_det/hico_agent_optimizer.py"
DB_ROOT="/tmp/vdms_gpbo_cB_s200_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_gpbo_cB_s200_${SLURM_JOB_ID}.json"
RESULTS_BASE="/work/hdd/bdjd/vdms_workflow/semantic_vdms/supplementary/results/hico_det_condB"
OUTPUT="${RESULTS_BASE}/gpbo/seed200.json"

echo "===== HICO-DET GP-BO ConditionB seed=200 ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "${RESULTS_BASE}/gpbo"

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "$PORT" > "$VDMS_CFG"

start_vdms() {
    apptainer instance stop "$INSTANCE" 2>/dev/null
    rm -rf "$DB_ROOT" && mkdir -p "$DB_ROOT"
    apptainer instance start --no-init --nv \
        --bind "${DB_ROOT}:/db" "$CONTAINER" "$INSTANCE"
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "$VDMS_CFG" \
        > /tmp/vdms_gpbo_cB_s200_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port $PORT"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID

start_vdms

PYTHONUNBUFFERED=1 \
    $PYTHON "$OPTIMIZER" \
        --port "$PORT" \
        --dataset-dir "$DATASET_DIR" \
        --method gp_bo \
        --iterations 50 \
        --seed 200 \
        --output "$OUTPUT" \
        --patience 0 \
        --clip-backbone vitl14 \
        --use-all-verb-ids \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"
echo "Finished: $(date)  Exit: $STATUS  Output: $OUTPUT"
exit $STATUS
