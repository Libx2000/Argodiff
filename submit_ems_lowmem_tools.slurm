#!/bin/bash
#SBATCH --job-name=ems-mem-tools
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-lowmem-tools-%j.out
#SBATCH --error=/fs6/home/yangjh16/zs_yangjh16/diffArgo/runs/ems-lowmem-tools-%j.err
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
action="${1:-status}"
phase="${2:-calibrate}"
config="${3:-configs/ems_v1_legacy.json}"
python -u ems_lowmem.py init --config "$config"
if [[ "$action" == smoke ]]; then
    python -u ems_lowmem.py run --config "$config" --phase "$phase" --task "${4:-4}" --smoke
elif [[ "$action" == freeze-final ]]; then
    echo 'No automatic unseen-year attestation. Confirm source/QC and scientific gates first.'
    exit 2
else
    python -u ems_lowmem.py "$action" --config "$config" --phase "$phase"
fi
