#!/bin/bash
# Submit the 6-run component-ablation sweep (2 conditions x 3 seeds).
#
# The OpenRouter key must be exported in THIS shell first -- it is forwarded to
# the jobs via sbatch's default --export=ALL and is never written to any file:
#
#     export OPENROUTER_API_KEY=...
#     bash submit_ghint.sh
#
set -u
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (it is not stored on disk)}"

cd "$(dirname "$0")"
mkdir -p logs

port=55730
for cond in noguid nohint; do
    for seed in 42 99 200; do
        name="hico_ghint_${cond}_s${seed}"
        out=$(COND=$cond SEED=$seed PORT=$port sbatch --job-name="$name" run_ghint_seed.sh 2>&1)
        echo "  ${name}  port=${port}  -> ${out}"
        port=$((port + 1))
    done
done
