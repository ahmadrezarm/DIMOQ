#!/bin/bash

#SBATCH --job-name=prune-s-m
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --mail-user=ahmad.rm0067@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Activate the virtual environment.
source /project/6101774/ahmadrm/virtual-environments/dimoq_env/bin/activate

# Define the log directory.
LOGDIR="/home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/hpc/results-paper"

# Prune the results.
python3 /home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/prune_results.py \
--log-dir "$LOGDIR" \
--seed 1 2 3 4 5 \
--env small medium \
--prune dds cdds pf ch
