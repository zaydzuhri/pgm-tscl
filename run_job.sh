#!/bin/bash
# Generic SLURM wrapper: everything after the script name is passed to `uv run`.
# Submit from the repo root:
#   sbatch run_job.sh train.py --config configs/sft.json
#   sbatch -J tscl_window run_job.sh train.py --config configs/tscl_window.json --seed 1
#   sbatch --time=00:30:00 --gpus-per-task=a100l:1 run_job.sh train.py --config configs/smoke.json
# sbatch flags (before run_job.sh) override the #SBATCH defaults below.
#SBATCH --job-name=pgm-tscl
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=248G
#SBATCH --gpus-per-task=a100l:1 
## Or --gpus-per-task=a100l:1    for an 80GB A100
## Or --gpus-per-task=1          for any GPU model (avoid rtx8000/v100: no native bf16)
#SBATCH --time=12:00:00

if [ $# -eq 0 ]; then
    echo "usage: sbatch run_job.sh <args for uv run>, e.g. train.py --config configs/sft.json"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Mila storage rules (docs.mila.quebec/technical_reference/clusters/mila/storage/):
# $HOME is small (100GB) and slow, for code. $SCRATCH (5TB, files unused for 90 days get deleted)
# is for re-downloadable data like the HuggingFace model/dataset cache.
export HF_HOME="${HF_HOME:-$SCRATCH/cache/huggingface}"

# wandb: uses `wandb login` (~/.netrc) if you have it, otherwise WANDB_API_KEY from .env (see .env.example).
# Turn wandb off for a run with: train.py ... --wandb_project ""
if [ -f .env ]; then set -a; source .env; set +a; fi
# Without this, Python buffers print() output when it goes to a file, so the SLURM log looks stuck.
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
echo "command: uv run $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run "$@"
