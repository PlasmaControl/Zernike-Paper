#!/bin/bash
#SBATCH --job-name=zernike-plots-cpu        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.9
conda activate desc-env

python create_plots_cpu.py
