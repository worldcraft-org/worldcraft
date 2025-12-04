#!/bin/bash

FOLDERNAME = $1

#SBATCH --job-name=semantic-nerf
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:3
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=%j_train_job.log

# process data
./run-nerfstudio.sh ns-process-data images --data $HOME/$FOLDERNAME/ --output-dir $HOME/processed_data/$FOLDERNAME

# train semantic-nerfw
./run-nerfstudio.sh ns-train semantic-nerfw --data $HOME/processed_data/$FOLDERNAME --output-dir $HOME/outputs/$FOLDERNAME --viewer.quit-on-train-completion True --pipeline.datamanager.pixel-sampler.ignore-mask True --max-num-iterations 7500

# export point cloud & find the latest run automatically
LATEST_RUN=$(ls -td $HOME/output/$FOLDERNAME/semantic-nerfw/* | head -1)
./run-nerfstudio.sh ns-export pointcloud --load-config ${LATEST_RUN}/config.yml --output-dir $HOME/export/$FOLDERNAME --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False
