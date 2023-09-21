#!/bin/bash

#SBATCH --chdir=/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/autostainer_study/
# // SBATCH --nodelist gv01
#SBATCH --time=8:00:00
#SBATCH --job-name=Run
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=qdp-alpha
#SBATCH --partition=v100_12
#SBATCH --gpu_cmode=shared
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out 

source ~/.bashrc
cd /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/autostainer_study/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

conda activate base
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

"$@"
