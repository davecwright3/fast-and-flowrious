#!/usr/bin/env bash
#SBATCH --job-name=freespec-mcmc
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=06:00:00
#SBATCH --partition=dri.q,hazboun.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

MODEL="${1:?usage: sbatch slurm/02-freespec-mcmc.sh <powerlaw|gaussian>}"

echo "Free-spectrum MCMC: model=$MODEL, realization=$SLURM_ARRAY_TASK_ID"

pixi run --as-is python scripts/02_freespec_mcmc.py freespec-mcmc \
    --sim-model "$MODEL" \
    --realization "$SLURM_ARRAY_TASK_ID"
