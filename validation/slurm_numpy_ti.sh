#!/bin/bash
# Production-length NumPy reference TI: 3 states x 2 replicas (array of 6 jobs, about 15 minutes each).
# Edit the account line before use; do not commit your allocation account to a public repository.
#SBATCH -A YOUR_ALLOCATION_ACCOUNT
#SBATCH -J RPMREF
#SBATCH -p cpu
#SBATCH --array=1-6
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o logs/RPMREF_%A_%a.out

ml Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source ~/jax-env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

STATES=(0.03 0.03 0.3 0.3 2.5 2.5)
SEEDS=(1 2 1 2 1 2)
i=$((SLURM_ARRAY_TASK_ID-1))
python validation/run_validation.py results/numpy_ti_${STATES[$i]}M_seed${SEEDS[$i]}.json \
    --production --c=${STATES[$i]} --seed=${SEEDS[$i]}
