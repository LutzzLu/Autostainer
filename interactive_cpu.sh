#!/bin/bash

source ~/.bashrc
conda activate base
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Other parameters:
# --nodes=1
# --ntasks-per-node=1
# --gpu_cmode=shared
# --nodelist=p03
# --gres=gpu:1
# --gpu_cmode=shared

srun \
--account qdp-alpha \
--job-name=interactive_cpu \
--partition=v100_12 \
--mem=256G \
--time=6:00:00 "$@"