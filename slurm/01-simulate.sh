#!/usr/bin/env bash
#SBATCH --job-name=simulate
#SBATCH --array=0-49%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --partition=hazboun.q,dri.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

MODEL="${1:?usage: sbatch slurm/01-simulate.sh <powerlaw|gaussian>}"

pixi run --as-is python scripts/01_simulate.py simulate \
    --model "$MODEL" \
    --realization "$SLURM_ARRAY_TASK_ID"
