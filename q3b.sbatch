#!/bin/bash
#SBATCH --job-name=q3b   # job name
#SBATCH --output=q3b.out # output log file
#SBATCH --error=q3b.err  # error file
#SBATCH --time=00:05:00  # 5 minutes of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

module load cuda

python ./q3.py> ./q3b.out
