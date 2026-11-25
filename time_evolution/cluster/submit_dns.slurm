#!/usr/bin/env bash
#SBATCH --job-name=pscalar_dns
#SBATCH --account=tii
#SBATCH --partition=sim
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-16%16
#SBATCH --output=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.out
#SBATCH --error=/nfs/users/stefano.pisoni/pscalar_runs/slurm-%x-%A_%a.err

# Scalar advected by a frozen DNS velocity snapshot (config 00059), at 512^2.
# The streamfunction is obtained by inverting the vorticity spectrally at 4096^2
# and coarsening, then compressed at psi_eps and differentiated with the
# solver's own MPOs so the velocity is divergence-free to 2e-12 discretely.
#
# psi_eps = 1e-2 is a modelling choice forced by the data: a real 2-D turbulent
# streamfunction saturates the geometric rank bound at eps = 1e-6 and then costs
# more parameters than the dense field. At 1e-2 it carries bond 21 and the
# derived velocity bond 34, against bond 4 for the cellular flow, which is why a
# step here costs several times more.
#
# kappa = 1.5e-2 puts the Batchelor scale at ~3 cells for the measured strain
# rate of 11. T = 2 is about 22 eddy turnover times.
#
#   sbatch --array=1-16 submit_dns.slurm                 # chi in {16,24,32,48}
#   sbatch --array=...  submit_dns.slurm "64 96 128"     # the extension
#
# The first sweep left the standard encoding at L2 = 6.6e-2 even at chi = 48, so
# the budget saving could be quoted at L2 = 1e-1 only. Its error falls as
# chi^-1.4, so chi = 64, 96 and 128 extend the overlap to ~2e-2. Wall time grows
# as chi^2 (N=0: 15 h at chi = 48), so chi = 96 fits in sim and N=0 chi = 128
# goes to long_sim. At N = 12 the head bond is capped at 2^(2L-N) = 64, so
# chi >= 64 is lossless there and is skipped (indices 1-9 for "64 96 128").

set -euo pipefail
readonly CHIS=${1:-"16 24 32 48"}

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
    local N bd cap
    for N in 0 8 10 12; do
        # largest geometric bond at L = 9: 512, except 64 at N = 12 (lossless above)
        cap=$(( N == 12 ? 64 : 512 ))
        for bd in ${CHIS}; do (( bd < cap )) && echo "${N} ${bd}"; done
    done
}
mapfile -t JOBS < <(build_jobs)
read -r N BD <<< "${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"

printf 'job=%s_%s host=%s commit=%s L=9 N=%s chi=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${N}" "${BD}" \
    "$(date --iso-8601=seconds)"

cd "${REPO}/time_evolution"
srun --ntasks=1 --cpus-per-task=1 \
    "${ENVDIR}/bin/python" run_decaying_cellular.py \
    --L 9 --N "${N}" --bd "${BD}" --kappa 1.5e-2 --T 2 \
    --velocity dns --psi-eps 1e-2 \
    --samples 24 --snaps 3 --label "DNS9_N${N}_bd${BD}"

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
