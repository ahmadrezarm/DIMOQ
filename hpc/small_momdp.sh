#!/bin/bash

#SBATCH --job-name=small-momdp
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --mail-user=ahmad.rm0067@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Activate the virtual environment.
source /project/6101774/ahmadrm/virtual-environments/dimoq_env/bin/activate

# Define the log directory.
LOGDIR="/home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/hpc/results"

# Run the experiments.
python3 /home/ahmadrm/projects/aip-frudzicz/ahmadrm/projects/DIMOQ/experiments.py \
--log-dir "$LOGDIR" \
--seed 1 2 3 4 5 \
--env small \
--warmup 50000 \
--num-episodes 2000 \
--save \
--log-every 5000 \
--num-threads 1