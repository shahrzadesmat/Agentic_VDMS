#!/bin/bash
#SBATCH --job-name=nsga2_smoke
#SBATCH --account=bdjd-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/Agentic_VDMS/eci/nsga2_smoke_%j.log
#SBATCH --error=/path/to/Agentic_VDMS/eci/nsga2_smoke_%j.err
#
# Validates NSGA-II evolutionary-baseline mechanics against a synthetic oracle.
# No VDMS, no container, no dataset — seconds of compute.
# Run this BEFORE burning a 3-seed GPU campaign.
#
# Usage: sbatch run_nsga2_smoke.sh

set -u

PYTHON="/path/to/venv/bin/python"

echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Started: $(date)"
cd /path/to/Agentic_VDMS/eci

PYTHONUNBUFFERED=1 $PYTHON nsga2_smoke_test.py
STATUS=$?

echo "Finished: $(date)  Exit: $STATUS"
if [ $STATUS -eq 0 ]; then
    echo "ALL CHECKS PASSED — safe to run the real NSGA-II campaign."
else
    echo "CHECKS FAILED — do not launch the campaign until resolved."
fi
exit $STATUS
