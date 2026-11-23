#!/bin/bash
# Production-length JAX thermodynamic integration: one job per state, 2 replicas each (vmapped).
# Run the short smoke test first to measure speed (see validation/README.md).
# Edit the account line before use; do not commit your allocation account to a public repository.
#SBATCH -A YOUR_ALLOCATION_ACCOUNT
#SBATCH -J RPMJAX
#SBATCH -p cpu
#SBATCH --array=1-3
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o logs/RPMJAX_%A_%a.out

ml Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source ~/jax-env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

STATES=(0.03 0.3 2.5)
python validation/run_validation_jax.py results/jax_ti_${STATES[$((SLURM_ARRAY_TASK_ID-1))]}M.json \
    --production --replicas 2 --c ${STATES[$((SLURM_ARRAY_TASK_ID-1))]}
