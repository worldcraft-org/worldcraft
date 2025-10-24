#!/bin/bash
# wrapper script to run nerfstudio container with gpu support on OCF HPC
# usage: ./run-nerfstudio.sh <command>
# example: ./run-nerfstudio.sh ns-train --help

# ensure nvidia libraries directory exists
if [ ! -d "$HOME/nvlibs" ]; then
    echo "copying nvidia libraries..."
    mkdir -p "$HOME/nvlibs"
    cp /usr/lib/x86_64-linux-gnu/libcuda.so.580.82.07 "$HOME/nvlibs/libcuda.so.1"
    cp /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.82.07 "$HOME/nvlibs/libnvidia-ml.so.1"
    echo "nvidia libraries copied to ~/nvlibs"
fi

# run the container with --nv flag and bind nvidia libraries
singularity exec --nv \
  --bind "$HOME/nvlibs:/usr/local/nvidia/lib64" \
  ~/nerfstudio_latest.sif "$@"
