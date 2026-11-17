#!/bin/bash
#SBATCH --job-name=ems-val61
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-val61-%j.out
#SBATCH --error=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-val61-%j.err
set -euo pipefail
cd /fs6/home/yangjh16/zs_yangjh16/diffArgo
source /fs6/home/yangjh16/miniconda3/etc/profile.d/conda.sh
conda activate diffArgo
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
action="${1:-all}"
policy="${2:-configs/ems_validate61_v1.json}"
printf 'ENTRY=ems-validate61-20261025-v1 JOB_ID=%s ACTION=%s\n' "${SLURM_JOB_ID:-unavailable}" "$action"
exec python -B -u ems_validate61.py "$action" --policy "$policy"
