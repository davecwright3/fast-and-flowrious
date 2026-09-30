#!/usr/bin/env bash
#SBATCH --job-name=postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=dri.q,hazboun.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j_err.log

pixi run --as-is python scripts/08_postprocess.py all
