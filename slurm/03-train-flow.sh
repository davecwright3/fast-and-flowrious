#!/usr/bin/env bash
#SBATCH --job-name=train-flow
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=dri.q,hazboun.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

MODEL="${1:?usage: sbatch slurm/03-train-flow.sh <powerlaw|gaussian>}"

echo "Training posterior flow: model=$MODEL, realization=$SLURM_ARRAY_TASK_ID"

pixi run --as-is python scripts/03_train_flow.py train-posterior-flow \
    --sim-model "$MODEL" \
    --realization "$SLURM_ARRAY_TASK_ID"
