#!/bin/bash
#SBATCH --job-name=cremspace_prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#
# sbatch --export=ALL,DB=/path/crem_v2.db,RADIUS=5,OUT=/path/slice_r5.db examples/slurm/prepare.sh
#   optional: SETS=a,b  EXTRA="--no-cycle --min-freq 2"  ENV=cremspace
set -e
: ${DB:?set DB} ${RADIUS:?set RADIUS} ${OUT:?set OUT}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV:-cremspace}
export TMPDIR=${TMPDIR:-/tmp}/cremspace_prepare_$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cremspace prepare -d "$DB" -r "$RADIUS" -o "$OUT" ${SETS:+--sets $SETS} --tmp-dir "$TMPDIR" \
    -c $SLURM_CPUS_PER_TASK --cache-mb 32768 $EXTRA
