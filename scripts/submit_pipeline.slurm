#!/bin/bash
#SBATCH --job-name=diffargo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=diffargo-%j.out
#SBATCH --error=diffargo-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${CONDA_BASE:-}" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${DIFFARGO_ENV:-diffargo}"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
stage="${1:?Choose direct, rollout, residual_stats, residual, or residual_rollout}"
shift
case "$stage" in
  direct) python -u train_direct.py --config configs/hpc_direct.yaml "$@" ;;
  rollout) python -u train_rollout.py --config configs/hpc_rollout_h7.yaml "$@" ;;
  residual_stats) python -u compute_residual_stats.py --config configs/residual_diffusion.yaml --deterministic-checkpoint runs/hpc_rollout_h7/checkpoint_best.pt --output artifacts/residual_stats_rollout_h7.npz "$@" ;;
  residual) python -u train_residual_diffusion.py --config configs/residual_diffusion.yaml --deterministic-checkpoint runs/hpc_rollout_h7/checkpoint_best.pt --residual-stats artifacts/residual_stats_rollout_h7.npz --output-dir runs/residual_diffusion_guided --epochs 40 --batch-size 1 --num-workers 4 "$@" ;;
  residual_rollout) python -u train_residual_rollout.py "$@" ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
