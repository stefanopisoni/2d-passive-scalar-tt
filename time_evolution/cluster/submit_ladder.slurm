#!/usr/bin/env bash
#SBATCH --job-name=pscalar_ladder
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
# One core per task, wide array. Extra threads do not merely fail to help this
# workload, they actively hurt it: 4 threads instead of 1 measured 29x SLOWER
# per step on these nodes. %32 leaves plenty of the partition free.
#SBATCH --array=1-44%32
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# Hybrid-vs-standard TT ladder at L=10 and L=11, as a job array: the runs are
# completely independent, so one array task per (L, N, chi) is the natural unit
# and the whole ladder finishes in the time of its slowest member.
#
# kappa ~ dx^2 keeps ell_B/dx fixed, so each L resolves the same range of scales
# and a finer grid adds structure rather than over-resolving what is there.
# Anchored at kappa = 5e-3 for L = 8.

set -euo pipefail

readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs

export PSCALAR_RESULTS="${RUN_ROOT}/results"
mkdir -p "${PSCALAR_RESULTS}"

# Pin every threading layer to 1. See the note in run_decaying_cellular.py:
# oversubscribing tiny operations is catastrophic here, not merely unhelpful.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

# One line per array task: L N chi kappa samples
build_jobs() {
    local L N bd kappa samples
    for L in 10 11 12; do
        if [[ "${L}" == "10" ]]; then
            kappa=3.125e-4; samples=40
            for N in 0 8 10 12; do for bd in 16 24 32 48 64; do
                echo "${L} ${N} ${bd} ${kappa} ${samples}"
            done; done
        elif [[ "${L}" == "11" ]]; then
            # fewer samples: each one recompresses a 2048^2 field three times
            kappa=7.8125e-5; samples=20
            for N in 0 10 12; do for bd in 24 32 48 64; do
                echo "${L} ${N} ${bd} ${kappa} ${samples}"
            done; done
        else
            # 4096^2. Reachable because a FIXED ACCURACY needs chi ~ 2^(0.34 L),
            # nearly flat, not the chi ~ 2^L that holding rho fixed would demand.
            # Measured single-threaded: 7.3 h at chi=24, 13.3 h at chi=32.
            kappa=1.953125e-5; samples=12
            for N in 0 10 12; do for bd in 16 24 32 48; do
                echo "${L} ${N} ${bd} ${kappa} ${samples}"
            done; done
        fi
    done
}

mapfile -t JOBS < <(build_jobs)
readonly LINE="${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read -r L N BD KAPPA SAMPLES <<< "${LINE}"

printf 'job=%s_%s host=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(date --iso-8601=seconds)"
printf 'repo=%s commit=%s L=%s N=%s chi=%s kappa=%s cpus=%s\n' \
    "${REPO}" "$(git -C "${REPO}" rev-parse --short HEAD)" \
    "${L}" "${N}" "${BD}" "${KAPPA}" "${SLURM_CPUS_PER_TASK}"

cd "${REPO}/time_evolution"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${ENVDIR}/bin/python" run_decaying_cellular.py \
    --L "${L}" --N "${N}" --bd "${BD}" --kappa "${KAPPA}" \
    --T 8 --samples "${SAMPLES}" --snaps 3 \
    --label "L${L}_N${N}_bd${BD}"

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
