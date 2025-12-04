#!/bin/bash

# SLURM directives (ignored when run standalone)
#SBATCH --job-name=semantic-nerf
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=%j_train_job.log

set -e

SCENE_NAME=$1
DATA_DIR=$2
OUTPUT_DIR=$3
EXPORT_DIR=$4

if [ -z "$SCENE_NAME" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$EXPORT_DIR" ]; then
    echo "Usage: $0 <scene_name> <data_dir> <output_dir> <export_dir>"
    echo "Example: $0 my_scene ./data/my_scene ./outputs ./exports"
    exit 1
fi

echo "=========================================="
echo "Semantic NeRF Training Pipeline"
echo "=========================================="
echo "Scene name: $SCENE_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Export directory: $EXPORT_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EXPORT_DIR"

# Process data
echo ""
echo "Step 1: Processing data with COLMAP..."
ns-process-data images \
    --data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/processed/$SCENE_NAME"

# Copy semantics to processed folder
echo ""
echo "Step 2: Copying semantic annotations..."
if [ -d "$DATA_DIR/semantics" ]; then
    cp -r "$DATA_DIR/semantics" "$OUTPUT_DIR/processed/$SCENE_NAME/"
    echo "✓ Semantics copied"
else
    echo "⚠️  WARNING: No semantics folder found in $DATA_DIR"
fi

# Copy panoptic_classes.json if it exists
if [ -f "$DATA_DIR/panoptic_classes.json" ]; then
    cp "$DATA_DIR/panoptic_classes.json" "$OUTPUT_DIR/processed/$SCENE_NAME/"
    echo "✓ panoptic_classes.json copied"
fi

# Train semantic-nerfw
echo ""
echo "Step 3: Training semantic-nerfw model..."
ns-train semantic-nerfw \
    --data "$OUTPUT_DIR/processed/$SCENE_NAME" \
    --output-dir "$OUTPUT_DIR/models/$SCENE_NAME" \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.pixel-sampler-num-rays-per-batch 4096

# Export point cloud
echo ""
echo "Step 4: Exporting point cloud..."
LATEST_RUN=$(ls -td "$OUTPUT_DIR/models/$SCENE_NAME/semantic-nerfw/"* | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No trained model found"
    exit 1
fi

echo "Using model: $LATEST_RUN"

ns-export pointcloud \
    --load-config "${LATEST_RUN}/config.yml" \
    --output-dir "$EXPORT_DIR/$SCENE_NAME" \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --save-world-frame True

echo ""
echo "=========================================="
echo "Training complete!"
echo "Point cloud: $EXPORT_DIR/$SCENE_NAME/point_cloud.ply"
echo "=========================================="

