#!/usr/bin/env bash
# Submit the whole pipeline as chained SLURM jobs, one array per stage.
#
#   01 simulate -> 02 freespec MCMC -> 03 train flow -> 04 fisher tails -> 05 NS -> 06 harmonic
#   01 simulate -> 07 full likelihood
#   everything  -> 08 postprocess
#
# Usage (from the repo root): ./slurm/launch-pipeline.sh [--dry-run]
set -e

MODELS=(powerlaw gaussian)

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

mkdir -p logs/

# submit <dependency> <job name> <script> [args...]; prints the job id.
submit() {
    local dep="$1" name="$2" script="$3"
    shift 3
    local args=(--parsable --job-name="$name")
    if [ -n "$dep" ]; then
        args+=(--dependency="$dep")
    fi
    if [ "$DRY_RUN" = 1 ]; then
        echo "sbatch ${args[*]} $script $*" >&2
        echo "DRY-$name"
        return
    fi
    sbatch "${args[@]}" "$script" "$@"
}

# aftercorr matches array indices, so every stage must be the full 0-49 array.
declare -A sim fs flow fish ns harm full
post_deps=""

# Per-model stages 01-04, plus the full-likelihood branch.
for m in "${MODELS[@]}"; do
    sim[$m]=$(submit "" "simulate-$m" slurm/01-simulate.sh "$m")
    echo "01 simulate  $m: ${sim[$m]}"

    full[$m]=$(submit "aftercorr:${sim[$m]}" "full-likelihood-$m" slurm/07-full-likelihood.sh "$m")
    echo "07 full-lik  $m: ${full[$m]}"
    post_deps="$post_deps:${full[$m]}"

    fs[$m]=$(submit "aftercorr:${sim[$m]}" "freespec-mcmc-$m" slurm/02-freespec-mcmc.sh "$m")
    echo "02 freespec  $m: ${fs[$m]}"

    flow[$m]=$(submit "aftercorr:${fs[$m]}" "train-flow-$m" slurm/03-train-flow.sh "$m")
    echo "03 flow      $m: ${flow[$m]}"

    fish[$m]=$(submit "aftercorr:${flow[$m]}" "fisher-tails-$m" slurm/04-fisher-tails.sh "$m")
    echo "04 fisher    $m: ${fish[$m]}"
done

# NS and harmonic for every (sim model, test model) pair.
for s in "${MODELS[@]}"; do
    for t in "${MODELS[@]}"; do
        ns[$s,$t]=$(submit "aftercorr:${fish[$s]}" "ns-$s-data-$t" slurm/05-nested-sampling.sh "$s" "$t")
        echo "05 NS        $s x $t: ${ns[$s,$t]}"

        harm[$s,$t]=$(submit "aftercorr:${ns[$s,$t]}" "harmonic-$s-data-$t" slurm/06-harmonic-evidence.sh "$s" "$t")
        echo "06 harmonic  $s x $t: ${harm[$s,$t]}"

        post_deps="$post_deps:${ns[$s,$t]}:${harm[$s,$t]}"
    done
done

# afterok: waits for every job in every array to finish.
post=$(submit "afterok${post_deps}" "postprocess" slurm/08-postprocess.sh)
echo "08 postprocess: $post"
