#!/usr/bin/env bash
#SBATCH --job-name=full-likelihood
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --partition=hazboun.q,dri.q
#SBATCH --account=hazboun
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --error=logs/%x-%A-%a_err.log

SIM="${1:?usage: sbatch slurm/07-full-likelihood.sh <sim_model>}"
RZ="$SLURM_ARRAY_TASK_ID"
OUTD="${FMC_OUTPUT_DIR:-outputs}/full_likelihood/$SIM/realization-$RZ"
echo "Full-likelihood BF: sim=$SIM realization=$RZ"

if [ -f "$OUTD/gauss_harmonic_bf.npz" ]; then
    echo "already done, skipping"; exit 0
fi
for TM in powerlaw gaussian; do
    if [ ! -f "$OUTD/chain_${TM}.npz" ]; then
        pixi run --as-is python scripts/07_full_likelihood.py sample \
            --sim-model "$SIM" --realization "$RZ" --test-model "$TM" || exit 1
    fi
done
pixi run --as-is python scripts/07_full_likelihood.py evidence \
    --sim-model "$SIM" --realization "$RZ"
