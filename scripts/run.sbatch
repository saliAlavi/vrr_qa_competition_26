#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/scratch/PAS2301/alialavi/vrr_qa_competition_26/logs/slurm_%x_%j.out
# Usage: sbatch [--gpus-per-node=N --time=HH:MM:SS] run.sbatch <args to run_eval.py>
set -uo pipefail
PROJ=/users/PAS2301/alialavi/projects/vrr_qa_competition_26
SCRATCH=/fs/scratch/PAS2301/alialavi
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

export PYTHONPATH=$PROJ/src
export HF_HOME=$SCRATCH/cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=$SCRATCH/cache/vllm
export TRITON_CACHE_DIR=$SCRATCH/cache/triton
export XDG_CACHE_HOME=$SCRATCH/cache/xdg
export WANDB_DIR=$SCRATCH/vrr_qa_competition_26/runs
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

echo "NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
$PY -u $PROJ/scripts/run_eval.py "$@"
echo "RUN_SBATCH_DONE"
