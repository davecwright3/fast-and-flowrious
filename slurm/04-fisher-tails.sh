#!/usr/bin/env bash
#SBATCH --job-name=fisher-tails
#SBATCH --array=0-49%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --partition=dri.q,hazboun.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

MODEL="${1:?usage: sbatch slurm/04-fisher-tails.sh <powerlaw|gaussian>}"

echo "Building fisher tails: model=$MODEL, realization=$SLURM_ARRAY_TASK_ID"

pixi run --as-is python scripts/04_build_fisher_tails.py run \
    --sim-model "$MODEL" \
    --realization "$SLURM_ARRAY_TASK_ID"
