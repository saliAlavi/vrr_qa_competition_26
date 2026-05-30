#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/fs/scratch/PAS2301/alialavi/vrr_qa_competition_26/logs/slurm_dp_%x_%j.out
# Data-parallel inference: one full model per GPU, each on a data shard, then merge.
# Usage: sbatch --gpus-per-node=N run_dp.sbatch <SPLIT> <RUN_ID> <args to run_eval.py...>
# e.g.   sbatch --gpus-per-node=4 run_dp.sbatch val q3vl8dec_dp --model qwen3-vl-8b --mode decompose
set -uo pipefail
PROJ=/users/PAS2301/alialavi/projects/vrr_qa_competition_26
SCRATCH=/fs/scratch/PAS2301/alialavi
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PYTHONPATH=$PROJ/src HF_HOME=$SCRATCH/cache HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=$SCRATCH/cache/vllm TRITON_CACHE_DIR=$SCRATCH/cache/triton XDG_CACHE_HOME=$SCRATCH/cache/xdg
export TOKENIZERS_PARALLELISM=false
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

SPLIT=$1; RUN_ID=$2; shift 2
N=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo "NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES  N=$N  RUN_ID=$RUN_ID"
nvidia-smi -L
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY -u $PROJ/scripts/run_eval.py \
    --split "$SPLIT" --run-id "$RUN_ID" --num-shards "$N" --shard-id "$i" --no-wandb "$@" \
    > $SCRATCH/vrr_qa_competition_26/logs/dp_${RUN_ID}_sh${i}.log 2>&1 &
done
wait
$PY -u $PROJ/scripts/merge_shards.py --run-id "$RUN_ID" --split "$SPLIT"
echo "RUN_DP_DONE"
