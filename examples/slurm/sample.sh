#!/bin/bash
#SBATCH --job-name=cremspace_sample
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=74:00:00
#
# Walkers for one or more sets; resubmit the same job to continue until `cremspace diagnose RUN` says DONE.
# One CPU per walker (WALKERS x number of sets), ~2 GB per walker plus the slice size.
# sbatch --export=ALL,SLICE=/path/slice_r5.db,RUN=/path/runs/r5_grow,MODE=grow,SETS=a,b examples/slurm/sample.sh
#   optional: WALKERS=16  HOURS=72 (keep below --time)  ENV=cremspace
set -e
: ${SLICE:?set SLICE} ${RUN:?set RUN}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV:-cremspace}
cremspace sample "$SLICE" -o "$RUN" --mode ${MODE:-grow} ${SETS:+--sets $SETS} \
    --walkers ${WALKERS:-16} --hours ${HOURS:-72}
cremspace diagnose "$RUN" -c 4 || true
