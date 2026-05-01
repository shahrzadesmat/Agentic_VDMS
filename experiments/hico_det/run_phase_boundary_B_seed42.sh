#!/bin/bash
#SBATCH --job-name=hico_phase_B
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/hico_det/hico_phase_B_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/hico_det/hico_phase_B_%j.err

# Phase boundary sensitivity Config B: late exploitation (0.50/0.80)
# N=50: exp=round(0.50*50)=25, expl=round(0.80*50)=40, fine=10 iterations

export OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY

SEED=42
PORT=55612
INSTANCE="vdms_phase_B_${SLURM_JOB_ID}"
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
DB_ROOT="/tmp/vdms_phase_B_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_phase_B_${SLURM_JOB_ID}.json"
FINAL_RESULTS="${BASE_DIR}/hicodet_finalized/ablations"
OUTPUT="${FINAL_RESULTS}/phase_boundary_B_seed42.json"

echo "====================================================="
echo "HICO-DET Phase Boundary Sensitivity — Config B"
echo "  t_exp_frac=0.50  t_expl_frac=0.80  (late exploitation)"
echo "  N=50: exp=25, expl=15, fine=10 iterations"
echo "  Seed: $SEED  Port: $PORT"
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Started: $(date)"
echo "====================================================="

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
        /vdms/build/vdms -cfg "$VDMS_CFG" \
        > /tmp/vdms_phase_B_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: VDMS did not become ready within 180 seconds"
        cat /tmp/vdms_phase_B_${SLURM_JOB_ID}.log
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
    --t-exp-frac 0.50 \
    --t-expl-frac 0.80 \
    --vdms-restart-cmd "bash -c start_vdms"
STATUS=$?

echo "Finished: $(date)  Exit: $STATUS"
echo "Results: $OUTPUT"

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT" "$VDMS_CFG"

exit $STATUS
