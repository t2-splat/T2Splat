#!/usr/bin/env bash
# Source this file: source /root/autodl-fs/T2-Splat-main/scripts/activate_local.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this script: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-fs/conda-envs/t2splat || return
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export SETUPTOOLS_USE_DISTUTILS=stdlib
export TORCH_EXTENSIONS_DIR=/root/autodl-fs/.cache/t2splat/torch-extensions
source /root/autodl-fs/T2-Splat-main/scripts/env.sh
