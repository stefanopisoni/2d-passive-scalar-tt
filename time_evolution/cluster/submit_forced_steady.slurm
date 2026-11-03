#!/usr/bin/env bash
#SBATCH --job-name=pscalar_fsteady
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --array=1-6%6
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# Forced cellular flow: the steady state by direct solve, and the approach to it
# by dense time-stepping, at 256^2, 512^2, 1024^2 with kappa ~ dx^2. Fixes T for
# the hybrid runs and gives each resolution its exact steady target.
#
# Probe lengths: the slowest mode decays on t ~ 53 at 256^2; T is set to cover
# ~15 such times if it scales like 1/kappa at 512^2, and the 1024^2 probe is
# sized to fit the partition's limits rather than an assumed scaling.

set -euo pipefail

readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs

export PSCALAR_RESULTS="${RUN_ROOT}/results/forced_steady"
mkdir -p "${PSCALAR_RESULTS}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"

JOBS=(
    "solve --L 8  --kappa 5e-3"
    "solve --L 9  --kappa 1.25e-3"
    "solve --L 10 --kappa 3.125e-4"
    "probe --L 8  --kappa 5e-3     --T 800"
    "probe --L 9  --kappa 1.25e-3  --T 800"
    "probe --L 10 --kappa 3.125e-4 --T 2400"
)
ARGS="${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"

printf 'job=%s_%s host=%s commit=%s args=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${ARGS}" \
    "$(date --iso-8601=seconds)"

cd "${REPO}/time_evolution"
# shellcheck disable=SC2086
srun --ntasks=1 --cpus-per-task=1 "${ENVDIR}/bin/python" forced_steady.py ${ARGS}

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
