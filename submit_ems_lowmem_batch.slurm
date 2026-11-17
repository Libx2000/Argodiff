#!/bin/bash
#SBATCH --job-name=ems-lowmem
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-lowmem-%j.out
#SBATCH --error=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-lowmem-%j.err
set -euo pipefail
cd /fs6/home/yangjh16/zs_yangjh16/diffArgo
source /fs6/home/yangjh16/miniconda3/etc/profile.d/conda.sh
conda activate diffArgo
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export EMS_DEADLINE="$(( $(date +%s) + 81000 ))"
phase="${1:-calibrate}"
first="${2:-0}"
last="${3:-7}"
config="${4:-configs/ems_v1_legacy.json}"
# init here creates ONLY the resource amendment, never replaces protocol.json.
python -u ems_lowmem.py init --config "$config"
python -u ems_lowmem.py batch --config "$config" --phase "$phase" --first "$first" --last "$last"
