#!/bin/bash
#SBATCH --job-name=ghint_smoke
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/Agentic_VDMS/eci/ghint_smoke_%j.log
#SBATCH --error=/path/to/Agentic_VDMS/eci/ghint_smoke_%j.err
#
# Verifies the --ablation-no-guidance / --ablation-no-hints flags are surgical.
# No VDMS, no dataset, no LLM API calls (requests.post is intercepted).
# Run this BEFORE launching the 6-run campaign.
#
# Usage: sbatch run_ghint_smoke.sh

set -u

PYTHON="/path/to/venv/bin/python"

echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Started: $(date)"
cd /path/to/Agentic_VDMS/src/hico_det

# Import the STAGED copy under src/hico_det -- that is the one the campaign runs,
# and only there does the module's own sys.path fix-up resolve vdtuner_ehvi.
PYTHONUNBUFFERED=1 PYTHONPATH="/path/to/Agentic_VDMS/src/hico_det:${PYTHONPATH:-}" \
    $PYTHON /path/to/Agentic_VDMS/eci/ghint_smoke_test.py
STATUS=$?

echo "Finished: $(date)  Exit: $STATUS"
if [ $STATUS -eq 0 ]; then
    echo "ALL CHECKS PASSED — ablations are surgical; safe to run the campaign."
else
    echo "CHECKS FAILED — do not launch the campaign until resolved."
fi
exit $STATUS
