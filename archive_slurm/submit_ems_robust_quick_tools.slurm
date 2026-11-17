#!/bin/bash
#SBATCH --job-name=ems-quick-tools
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-quick-tools-%j.out
#SBATCH --error=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-quick-tools-%j.err
set -euo pipefail
cd /fs6/home/yangjh16/zs_yangjh16/diffArgo
source /fs6/home/yangjh16/miniconda3/etc/profile.d/conda.sh
conda activate diffArgo
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
action="${1:-inspect}"
case "$action" in inspect|status|summarize) ;; *) printf '%s\n' 'Use inspect, status or summarize'; exit 2 ;; esac
exec python -B -u ems_robust_quick.py "$action"
