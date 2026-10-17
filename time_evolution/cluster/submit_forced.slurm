#!/usr/bin/env bash
#SBATCH --job-name=pscalar_forced
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-12%12
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# Cellular flow with a steady source, so that <theta^2> settles where production
# balances dissipation and the statistics describe a steady state rather than a
# decaying transient.
#
# The source is f = 0.8 sin(4y). Forcing at k=1 was measured not to equilibrate:
# the approach to steady state is set by the diffusive time of the forced scale,
# 1/(kappa k_f^2), which is 200 time units at k_f=1 and 12 at k_f=4. At k_f=4 the
# variance plateaus by t ~ 40 at 0.51, so T = 60 leaves a usable window.
#
# L=8 keeps 12245 steps affordable; this is a physics check on the rank result,
# not a resolution study.

set -euo pipefail

readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs

export PSCALAR_RESULTS="${RUN_ROOT}/results"
mkdir -p "${PSCALAR_RESULTS}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

build_jobs() {
    local N bd
    for N in 0 6 8 10; do for bd in 16 24 32; do echo "${N} ${bd}"; done; done
}
mapfile -t JOBS < <(build_jobs)
read -r N BD <<< "${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"

printf 'job=%s_%s host=%s commit=%s L=8 N=%s chi=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${N}" "${BD}" \
    "$(date --iso-8601=seconds)"

cd "${REPO}/time_evolution"
srun --ntasks=1 --cpus-per-task=1 \
    "${ENVDIR}/bin/python" run_decaying_cellular.py \
    --L 8 --N "${N}" --bd "${BD}" --kappa 5e-3 --T 60 \
    --forcing-amp 0.8 --forcing-k 4 \
    --samples 60 --snaps 4 --label "FRC8_N${N}_bd${BD}"

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
