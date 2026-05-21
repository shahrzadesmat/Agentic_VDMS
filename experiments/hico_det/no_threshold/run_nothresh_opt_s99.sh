#!/bin/bash
#SBATCH --job-name=nt_opt_s99
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# ── USER CONFIG ─────────────────────────────────────────────────────────────
# Edit the four variables below to match your environment before submitting.
# Also update --account and --partition in the #SBATCH header above.
BASE_DIR="/path/to/Agentic_VDMS"  # root of semantic_vdms/
DATASET_DIR="/path/to/datasets"              # dataset root
CONTAINER="/path/to/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/path/to/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

PORT=55668
INSTANCE="vdms_nothresh_nt_opt_s99_${SLURM_JOB_ID}"
SRC="/path/to/Agentic_VDMS/src/hico_det"
DB_ROOT="/tmp/vdms_nothresh_nt_opt_s99_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_nothresh_nt_opt_s99_${SLURM_JOB_ID}.json"
RESULTS="${BASE_DIR}/hico_det/no_threshold/results"
OUTPUT="${RESULTS}/nothresh_opt_s99.json"

echo "===== nt_opt_s99 (no-threshold ablation) ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Started: $(date)"

mkdir -p "${RESULTS}"
cd "${SRC}"

printf '{\n  "port": %d,\n  "db_root_path": "/db",\n  "max_simultaneous_clients": 100\n}' \
    "${PORT}" > "${VDMS_CFG}"

start_vdms() {
    apptainer instance stop "${INSTANCE}" 2>/dev/null || true
    rm -rf "${DB_ROOT}" && mkdir -p "${DB_ROOT}"
    apptainer instance start --no-init --nv \
        --bind "${DB_ROOT}:/db" "${CONTAINER}" "${INSTANCE}"
    apptainer exec instance://${INSTANCE} \
        /vdms/build/vdms -cfg "${VDMS_CFG}" \
        > /tmp/vdms_nothresh_nt_opt_s99_${SLURM_JOB_ID}.log 2>&1 &
    timeout 180 bash -c "until nc -z localhost ${PORT}; do sleep 1; done"
    echo "[VDMS] Ready on port ${PORT}"
}
export -f start_vdms
export INSTANCE DB_ROOT PORT CONTAINER VDMS_CFG SLURM_JOB_ID

start_vdms

PYTHONUNBUFFERED=1 \
    ${PYTHON} hico_agent_optimizer.py \
        --port "${PORT}" \
        --dataset-dir "${DATASET_DIR}" \
        --method optuna \
        --iterations 50 \
        --seed 99 \
        --output "${OUTPUT}" \
        --clip-backbone vitl14 \
        --patience 0 \
        --no-threshold \
        --vdms-restart-cmd "bash -c start_vdms"

STATUS=$?
apptainer instance stop "${INSTANCE}" 2>/dev/null || true
rm -rf "${DB_ROOT}" "${VDMS_CFG}"
echo "Finished: $(date)  Exit: ${STATUS}  Output: ${OUTPUT}"
exit ${STATUS}
