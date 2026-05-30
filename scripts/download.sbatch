#!/bin/bash
#SBATCH --account=pas2966
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/scratch/PAS2301/alialavi/datasets/implicitqa/logs/download_%j.out
set -uo pipefail
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PATH=/users/PAS2301/alialavi/miniconda3/envs/nips/bin:$PATH
export VIDEO_DIR=/fs/scratch/PAS2301/alialavi/datasets/implicitqa/videos
export DL_WORKERS=8
$PY -u /users/PAS2301/alialavi/projects/vrr_qa_competition_26/scripts/download_videos.py
echo DOWNLOAD_JOB_DONE
