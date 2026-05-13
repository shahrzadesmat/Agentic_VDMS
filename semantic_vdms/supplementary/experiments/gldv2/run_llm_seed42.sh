#!/bin/bash
#SBATCH --job-name=gldv2_agent
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --exclude=gpub066,gpub088
#SBATCH --output=/work/hdd/bdjd/vdms_workflow/semantic_vdms/gldv2/gldv2_llm_%j.log
#SBATCH --error=/work/hdd/bdjd/vdms_workflow/semantic_vdms/gldv2/gldv2_llm_%j.err

# LLM/Grid/Random/Optuna optimizer — GLDv2 Landmark Retrieval
# Usage: sbatch run_gldv2_agent.sh [method]
# method: hyperparameter_only (default), grid, random, or optuna
#
# HNSW index build at 762K takes ~4.5 min (batch_properties=500). 4h wall time is sufficient.

METHOD="${1:-hyperparameter_only}"
scontrol update JobId=$SLURM_JOB_ID JobName="gldv2_${METHOD}" 2>/dev/null
SEED=42
PORT=55596
INSTANCE="vdms_gldv2_agent"
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"
DB_ROOT="/tmp/vdms_gldv2_agent_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_gldv2_agent_${SLURM_JOB_ID}.json"
OUTPUT="gldv2/gldv2_${METHOD}_seed${SEED}_results_${SLURM_JOB_ID}.json"

echo "========================================="
echo "GLDv2 Optimizer — method=${METHOD}"
echo "  Seed:       $SEED"
echo "  Iterations: 30 (patience=0)"
echo "  Backbone:   CLIP ViT-L/14 (768-d)"
echo "  Score:      mAP x QPS"
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="

cd "$BASE_DIR/gldv2"

# Verify required GLDv2 feature files
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
        /vdms/build/vdms -cfg "$VDMS_CFG" > /tmp/vdms_gldv2_agent_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: VDMS did not become ready within 180 seconds"
        cat /tmp/vdms_gldv2_agent_${SLURM_JOB_ID}.log
        apptainer instance stop "$INSTANCE" 2>/dev/null; exit 1
    fi
    echo "[VDMS] Ready on port $PORT (fresh DB)"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID

start_vdms

PYTHONUNBUFFERED=1 $PYTHON gldv2_agent_optimizer.py \
    --port "$PORT" \
    --dataset-dir "$DATASET_DIR" \
    --method "$METHOD" \
    --iterations 30 \
    --seed "$SEED" \
    --output "$OUTPUT" \
    --model "minimax/minimax-m2.1" \
    --patience 0 \
    --vdms-restart-cmd "bash -c start_vdms"
STATUS=$?

echo "Finished: $(date)  Exit: $STATUS"
echo "Results: $OUTPUT"

apptainer instance stop "$INSTANCE" 2>/dev/null
rm -rf "$DB_ROOT"

exit $STATUS
