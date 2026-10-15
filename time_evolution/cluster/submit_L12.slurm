#!/usr/bin/env bash
#SBATCH --job-name=pscalar_L12
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=4-00:00:00
#SBATCH --array=1-12%12
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# The 4096^2 ladder. Submitted with a dependency on the dense-cache job, because
# every task would otherwise step the SAME dense reference, and at this size that
# costs 56.5 h -- far longer than the hybrid evolution it exists to validate.
# Building it once turns twelve ~60-hour jobs into twelve 5-to-25-hour ones and
# saves ~680 core-hours.

set -euo pipefail

readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs
readonly CACHE="${RUN_ROOT}/dense_L12_T8_s12"
readonly KAPPA=1.953125e-5
readonly SAMPLES=12

export PSCALAR_RESULTS="${RUN_ROOT}/results"
mkdir -p "${PSCALAR_RESULTS}"

# See run_decaying_cellular.py: oversubscribing these tiny operations is
# catastrophic, not merely unhelpful (29x slower per step at 4 threads).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

build_jobs() {
    local N bd
    for N in 0 10 12; do for bd in 16 24 32 48; do echo "${N} ${bd}"; done; done
}
mapfile -t JOBS < <(build_jobs)
read -r N BD <<< "${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"

printf 'job=%s_%s host=%s commit=%s L=12 N=%s chi=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${N}" "${BD}" \
    "$(date --iso-8601=seconds)"

cd "${REPO}/time_evolution"
srun --ntasks=1 --cpus-per-task=1 \
    "${ENVDIR}/bin/python" run_decaying_cellular.py \
    --L 12 --N "${N}" --bd "${BD}" --kappa "${KAPPA}" \
    --T 8 --samples "${SAMPLES}" --snaps 3 \
    --dense-cache "${CACHE}" --label "L12_N${N}_bd${BD}"

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
