#!/usr/bin/env bash
#SBATCH --job-name=nested-sampling
#SBATCH --array=0-49%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=dri.q,hazboun.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

SIM_MODEL="${1:?usage: sbatch slurm/05-nested-sampling.sh <sim_model> <test_model>}"
TEST_MODEL="${2:?usage: sbatch slurm/05-nested-sampling.sh <sim_model> <test_model>}"

echo "Nested sampling: sim_model=$SIM_MODEL, realization=$SLURM_ARRAY_TASK_ID, test_model=$TEST_MODEL"

pixi run --as-is python scripts/05_nested_sampling.py run-nested-sampling \
    --sim-model "$SIM_MODEL" \
    --realization "$SLURM_ARRAY_TASK_ID" \
    --test-model "$TEST_MODEL"
