#!/bin/bash
#SBATCH --job-name=zernike-plots        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.9
conda activate desc-env

python create_plots_gpu.py
