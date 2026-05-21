#!/bin/bash
#SBATCH --job-name=nt_rand_s200
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
BASE_DIR="/work/hdd/bdjd/vdms_workflow/semantic_vdms"  # root of semantic_vdms/
DATASET_DIR="/work/hdd/bdjd/vdms/datasets"              # dataset root
CONTAINER="/work/hdd/bdjd/vdms_latest.sif"              # Apptainer .sif image
PYTHON="/work/hdd/bdjd/vdms_code/venv/bin/python"       # Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

PORT=55666
INSTANCE="vdms_nothresh_nt_rand_s200_${SLURM_JOB_ID}"
SRC="/work/hdd/bdjd/vdms_workflow/src/hico_det"
DB_ROOT="/tmp/vdms_nothresh_nt_rand_s200_${SLURM_JOB_ID}"
VDMS_CFG="/tmp/vdms_nothresh_nt_rand_s200_${SLURM_JOB_ID}.json"
RESULTS="${BASE_DIR}/hico_det/no_threshold/results"
OUTPUT="${RESULTS}/nothresh_rand_s200.json"

echo "===== nt_rand_s200 (no-threshold ablation) ====="
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
        > /tmp/vdms_nothresh_nt_rand_s200_${SLURM_JOB_ID}.log 2>&1 &
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
        --method random \
        --iterations 50 \
        --seed 200 \
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
