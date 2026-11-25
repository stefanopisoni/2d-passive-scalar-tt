#!/usr/bin/env bash
#SBATCH --job-name=pscalar_kr
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

# Kraichnan passive scalar (kraichnan.py, run_kraichnan.py), white-in-time
# velocity and forcing, renewal discretisation with Kubo number ku.
#
#   sbatch --array=1 submit_kraichnan.slurm test
#   sbatch --array=1-8 submit_kraichnan.slurm valid
#   sbatch --array=1-N submit_kraichnan.slurm hold L XI "CHI..." [T] [budget|plain]
#
# FAMILY=test   tests/test_kraichnan.py.
# FAMILY=valid  dense runs to T=200, spectra averaged over t >= 20, compared
#               with the stationary lattice spectrum by plot_kraichnan.py.
#               Ku = 0.5 and 0.25 and CFL = 0.5 and 0.25 at 256^2 measure the
#               distance from the white-noise limit and the time-stepping
#               error; the ku=0.5 runs also save their final field, from which
#               the hold runs start.
# FAMILY=hold   hybrid runs at L, XI (2 or 43), started from that field with a
#               fresh noise seed; the array runs over N x CHI, N in {0,6,8,10}
#               at L = 8 and {0,8,10,12} at L = 9, skipping chi at or above the
#               chain's geometric bound (lossless). 120 samples, each recording
#               the paper's increment statistics for both fields, so the second
#               half averages 60. The task count is printed by
#                 bash submit_kraichnan.slurm count L "CHI..."

set -euo pipefail

readonly FAMILY=$1
if [[ "${FAMILY}" == count ]]; then
    L=$2
    for N in $([[ "${L}" == 8 ]] && echo "0 6 8 10" || echo "0 8 10 12"); do
        if (( N <= L )); then cap=$((1 << L)); else cap=$((1 << (2 * L - N))); fi
        for bd in $3; do (( bd < cap )) && echo "${N} ${bd}"; done
    done | awk '{print NR": N="$1" chi="$2}'
    exit 0
fi
readonly REPO=/nfs/users/stefano.pisoni/2d-passive-scalar-tt
readonly ENVDIR=/nfs/users/stefano.pisoni/.venvs/pscalar
readonly RUN_ROOT=/nfs/users/stefano.pisoni/pscalar_runs
readonly SPIN="${RUN_ROOT}/results/kraichnan/spin"

export PSCALAR_RESULTS="${RUN_ROOT}/results/kraichnan"
mkdir -p "${PSCALAR_RESULTS}" "${SPIN}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PSCALAR_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib-cache"
export NUMBA_CACHE_DIR="${RUN_ROOT}/numba-cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

readonly XI43=1.3333333333333333
PY="${ENVDIR}/bin/python"
cd "${REPO}/time_evolution"

printf 'job=%s_%s host=%s commit=%s family=%s args=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" \
    "$(git -C "${REPO}" rev-parse --short HEAD)" "${FAMILY}" "${*:2}" \
    "$(date --iso-8601=seconds)"

case "${FAMILY}" in
test)
    srun --ntasks=1 --cpus-per-task=1 "${PY}" tests/test_kraichnan.py
    ;;
valid)
    #        L  xi       ku    cfl   label                    save
    CASES=( "8  2        0.5   0.5   KV8_xi2_ku0.5            spin_L8_xi2"
            "8  2        0.25  0.5   KV8_xi2_ku0.25           -"
            "8  2        0.5   0.25  KV8_xi2_ku0.5_cfl0.25    -"
            "8  ${XI43}  0.5   0.5   KV8_xi43_ku0.5           spin_L8_xi43"
            "8  ${XI43}  0.25  0.5   KV8_xi43_ku0.25          -"
            "8  ${XI43}  0.5   0.25  KV8_xi43_ku0.5_cfl0.25   -"
            "9  2        0.5   0.5   KV9_xi2_ku0.5            spin_L9_xi2"
            "9  ${XI43}  0.5   0.5   KV9_xi43_ku0.5           spin_L9_xi43" )
    read -r L XI KU CFL LABEL SAVE <<< "${CASES[$((SLURM_ARRAY_TASK_ID - 1))]}"
    EXTRA=()
    [[ "${SAVE}" != "-" ]] && EXTRA=(--save-final "${SPIN}/${SAVE}.npy")
    srun --ntasks=1 --cpus-per-task=1 "${PY}" run_kraichnan.py \
        --L "${L}" --xi "${XI}" --ku "${KU}" --cfl "${CFL}" --dense \
        --T 200 --t-avg 20 --seed 1 --label "${LABEL}" "${EXTRA[@]}"
    ;;
hold)
    L=$2 XITAG=$3 CHIS=$4 T=${5:-20} POLICY=${6:-budget}
    # budget: every compressed update is rescaled to the variance budget
    # integrated by the stepper (labels KH); plain: plain truncation (labels KP)
    [[ "${POLICY}" == budget ]] && PFX=KH || PFX=KP
    XI=$([[ "${XITAG}" == 2 ]] && echo 2 || echo "${XI43}")
    # head sizes as in the forced sweeps: {0,6,8,10} at 256^2, {0,8,10,12} above
    HEADS=$([[ "${L}" == 8 ]] && echo "0 6 8 10" || echo "0 8 10 12")
    build_jobs() {
        local N bd cap
        for N in ${HEADS}; do
            # largest geometric bond of the chain; chi >= cap is lossless
            if (( N <= L )); then cap=$((1 << L)); else cap=$((1 << (2 * L - N))); fi
            for bd in ${CHIS}; do (( bd < cap )) && echo "${N} ${bd}"; done
        done
    }
    mapfile -t JOBS < <(build_jobs)
    read -r N BD <<< "${JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}"
    srun --ntasks=1 --cpus-per-task=1 "${PY}" run_kraichnan.py \
        --L "${L}" --xi "${XI}" --ku 0.5 --cfl 0.5 --N "${N}" --bd "${BD}" \
        --init "${SPIN}/spin_L${L}_xi${XITAG}.npy" --seed 2 --T "${T}" \
        --samples 120 --t-avg 0 --spec-every 0 --policy "${POLICY}" \
        --label "${PFX}${L}_xi${XITAG}_N${N}_bd${BD}"
    ;;
*)
    echo "unknown family ${FAMILY}" >&2; exit 2 ;;
esac

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
