#!/bin/bash

#SBATCH --job-name=soa_base_array
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=long,medium
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# ================================================================
# USER CONFIGURATION - only edit this block between studies
# ================================================================
# Every parametric study is this same script with 1-2 values changed.
# Set STUDY (the folder name under results/) and the knob(s) below, then
# launch with: sbatch submit_soa_base_cpar.sl
#
# Reproducing each study (STUDY  ->  what to change vs. this base):
#   soa_base       baseline, as written below
#   soa_nbin201    CAVITY_NBIN=201
#   soa_nbin10001  CAVITY_NBIN=10001  + single case: CASES=(goal),  --array=0-0
#   soa_10e5turns  N_TURNS=100000
#   soa_alphalin   USE_LINEAR_MCF=1
#   soa_elegantfreq                   single case: CASES=(goal_elegant_freq), --array=0-0
#
# NOTE: --array is an #SBATCH directive parsed before the shell runs, so for the
# single-case studies you MUST edit BOTH the CASES array AND the --array bound at
# the top of this file (set --array=0-0). The task-ID guard makes over-launching
# safe but wasteful.
export STUDY=soa_base
CASES=(no_hc hc_100kv goal flat_potential)
export N_TURNS=50000
export CAVITY_NBIN=1001
export USE_LINEAR_MCF=0
# ================================================================


## Bunch-length validation runs (no_hc, hc_100kv, goal, flat_potential), parallelised.
##
## Array tasks 0-3 map to the four validation cases. All run in parallel,
## each on its own node with 16 MPI ranks.
##
## To run a subset of cases, edit the CASES array in the USER CONFIGURATION block above.
##
## Usage:
##   sbatch submit_soa_base_cpar.sl
##
## Per-task logs: results/mbxlogs/soa_base_<arrayjobid>_<taskid>.txt
## Individual case logs: results/<study>/<study>_<case>/validation_<case>_<jobid>.log

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR}/results/mbxlogs"

exec > >(tee "${SLURM_SUBMIT_DIR}/results/mbxlogs/soa_base_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt") \
     2>&1

cd "${SLURM_SUBMIT_DIR}/src/simulation"

ml purge
ml OpenMPI/4.1.4-GCC-11.3.0-cxx
ml SciPy-bundle/2022.05-foss-2022a matplotlib/3.5.2-foss-2022a

VENV_DIR="${VENV_DIR:-$HOME/tfg/mbtrack2/.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -d "${SLURM_SUBMIT_DIR}/.venv" ]; then
    source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
else
    echo "No virtual environment found. Using system/module Python."
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

NPROC=${SLURM_NTASKS:-16}
if (( 448 % NPROC != 0 )); then
    echo "ERROR: number of MPI ranks must divide 448 exactly. Got NPROC=${NPROC}."
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#CASES[@]}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: no case assigned (${#CASES[@]} cases defined). Exiting."
    exit 0
fi
export VALIDATION_CASE=${CASES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "SLURM array job   : ${SLURM_ARRAY_JOB_ID}"
echo "Array task        : ${SLURM_ARRAY_TASK_ID}/3  ->  ${VALIDATION_CASE}"
echo "Host              : $(hostname)"
echo "MPI ranks         : ${NPROC}"
echo "time              : $(date)"
echo "============================================================"

export OUTDIR="${OUTDIR:-${SLURM_SUBMIT_DIR}/results/${STUDY}/${STUDY}_${VALIDATION_CASE}}"
mkdir -p "${OUTDIR}"

VALIDATION_LOG="${OUTDIR}/validation_${VALIDATION_CASE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

start_time=$(date +%s)

mpiexec -np "${NPROC}" \
    -x VALIDATION_CASE -x STUDY -x CAVITY_NBIN -x N_TURNS -x USE_LINEAR_MCF \
    python3 -u soasim.py \
    > >(tee "${VALIDATION_LOG}") \
    2> >(tee -a "${VALIDATION_LOG}" >&2)

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$(python3 -c "print(f'{${duration}/3600:.2f}')")

echo ""
echo "============================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} (${VALIDATION_CASE}) finished."
echo "Execution time: ${hours} hours"
echo "time: $(date)"
echo "============================================================"
