#!/bin/bash

source ~/.bashrc
conda activate base
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Other parameters:
# --nodes=1
# --ntasks-per-node=1
# --gpu_cmode=shared
# --nodelist=p03

srun \
--account qdp-alpha \
--job-name= \
--partition=v100_12 \
--gres=gpu:1 \
--gpu_cmode=shared \
--mem=32G \
--time=6:00:00 "$@"
