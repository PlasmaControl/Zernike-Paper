#!/bin/bash
#SBATCH --job-name=zernike-accuracy       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.9
conda activate desc-env

python create_accuracy_plot.py
