#!/usr/bin/env bash
#SBATCH --job-name=pscalar_fhires
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# Forced cellular flow at 512^2 and 1024^2, kappa ~ dx^2, f = 0.8 sin(4y).
#
#   sbatch --array=1-N submit_forced_hires.slurm FAMILY L "CHI..."
#
# FAMILY=hold    start from the exact dense steady state (forced_steady.py
#                solve). The slowest mode relaxes on t = 53, 208 at 256^2, 512^2
#                (~1/kappa), so ~830 at 1024^2: a spin-up from the random field
#                does not fit a 5-day job there, and a hold run measures the
#                truncation error at steady state directly.
#                T = 400 at 512^2 (~2 relaxation times), 200 at 1024^2.
# FAMILY=spinup  start from the usual low-k random field; T = 600 at 512^2,
#                where <theta^2> is within 1% of the steady value.
# FAMILY=holdlong  the hold run to T = 600 at 1024^2, for the cases whose error
#                was still growing at T = 200 (N=0 chi=16,24; N=12 chi=12,16).
#
# The array runs over N in {0, 8, 10, 12} x the given chi list.
#
# Measured cost per step on sim nodes at chi=32: 0.5-1.3 s at 512^2, 0.9-2.2 s
# at 1024^2 (N=12 the dearest). A hold run is 163k steps, so 1024^2 N=12 chi=32
# needs ~107 h and is submitted to long_sim; everything else fits 5 days.

set -euo pipefail

readonly FAMILY=$1 L=$2 CHIS=$3
readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs
readonly STEADY="${RUN_ROOT}/results/forced_steady/steady_L${L}.npy"

case "${L}" in
    9)  KAPPA=1.25e-3 ;;
    10) KAPPA=3.125e-4 ;;
    *)  echo "unsupported L=${L}" >&2; exit 2 ;;
esac
case "${FAMILY}:${L}" in
    hold:9)    T=400; INIT=(--init "${STEADY}"); TAG=HOLD ;;
    hold:10)   T=200; INIT=(--init "${STEADY}"); TAG=HOLD ;;
    spinup:9)  T=600; INIT=();                   TAG=FRC ;;
    holdlong:10) T=600; INIT=(--init "${STEADY}"); TAG=LONG ;;
    *)         echo "unsupported ${FAMILY} at L=${L}" >&2; exit 2 ;;
esac

export PSCALAR_RESULTS="${RUN_ROOT}/results/forced_hires"
mkdir -p "${PSCALAR_RESULTS}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

build_jobs() {
    local N bd
    for N in 0 8 10 12; do for bd in ${CHIS}; do echo "${N} ${bd}"; done; done
}
mapfile -t JOBS < <(build_jobs)
read -r N BD <<< "${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"
LABEL="${TAG}${L}_N${N}_bd${BD}"

printf 'job=%s_%s host=%s commit=%s %s L=%s N=%s chi=%s T=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${FAMILY}" "${L}" "${N}" \
    "${BD}" "${T}" "$(date --iso-8601=seconds)"

cd "${REPO}/time_evolution"
srun --ntasks=1 --cpus-per-task=1 \
    "${ENVDIR}/bin/python" run_decaying_cellular.py \
    --L "${L}" --N "${N}" --bd "${BD}" --kappa "${KAPPA}" --T "${T}" \
    --forcing-amp 0.8 --forcing-k 4 "${INIT[@]}" \
    --samples 60 --snaps 4 --label "${LABEL}"

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
