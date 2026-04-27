#!/bin/bash

#SBATCH --job-name=large-4-new
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
LOGDIR="/home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/hpc/results-large"

# Run the experiments.
python3 /home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/experiments.py \
--log-dir "$LOGDIR" \
--seed 4 \
--env large \
--warmup 50000 \
--num-episodes 2000 \
--save \
--log-every 5000 \
--num-threads 1