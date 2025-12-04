#!/usr/bin/env python3
"""
Worldcraft Pipeline Orchestrator

This script orchestrates the entire Worldcraft pipeline:
1. Semantic segmentation of images
2. NeRF training with semantic annotations
3. Point cloud export
4. Voxelization
5. Minecraft conversion

Usage:
    python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_directory_structure(scene_dir):
    """Validate input directory structure."""
    print("\n" + "=" * 70)
    print("Validating Input Directory")
    print("=" * 70)
    
    images_dir = os.path.join(scene_dir, "images")
    
    if not os.path.exists(scene_dir):
        print(f"❌ Error: Scene directory not found: {scene_dir}")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Error: Images directory not found: {images_dir}")
        print(f"\nExpected structure:")
        print(f"  {scene_dir}/")
        print(f"    images/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"      ...")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"❌ Error: No images found in {images_dir}")
        return False
    
    print(f"✓ Scene directory: {scene_dir}")
    print(f"✓ Found {len(image_files)} images in {images_dir}")
    return True


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: Command not found: {cmd[0]}")
        print("Make sure all dependencies are installed.")
        return False


def stage_1_semantic_segmentation(scene_dir):
    """Stage 1: Process images to generate semantic segmentation masks."""
    print("\n" + "#" * 70)
    print("# STAGE 1: Semantic Segmentation")
    print("#" * 70)
    
    process_script = "image-processing/process_semantics.py"
    
    # Check if semantics already exist
    semantics_dir = os.path.join(scene_dir, "semantics")
    if os.path.exists(semantics_dir):
        semantic_files = [f for f in os.listdir(semantics_dir) if f.endswith('.png')]
        if semantic_files:
            print(f"\n✓ Semantics already exist ({len(semantic_files)} files)")
            response = input("Regenerate semantic masks? (y/N): ")
            if response.lower() != 'y':
                print("Skipping semantic segmentation")
                return True
    
    # Pass scene directory as command-line argument
    cmd = ["python", process_script, "--scene-dir", scene_dir]
    return run_command(cmd, "Semantic Segmentation Processing")


def stage_2_nerf_training(scene_dir, output_dir, export_dir, scene_name):
    """Stage 2: Train semantic NeRF and export point cloud."""
    print("\n" + "#" * 70)
    print("# STAGE 2: Semantic NeRF Training & Point Cloud Export")
    print("#" * 70)
    
    # Check if model already exists
    model_dir = os.path.join(output_dir, "models", scene_name)
    if os.path.exists(model_dir):
        print(f"\n⚠️  Model directory already exists: {model_dir}")
        response = input("Retrain model? (y/N): ")
        if response.lower() != 'y':
            print("Skipping NeRF training")
            return True
    
    train_script = "semnerf/train_job.sh"
    
    if not os.path.exists(train_script):
        print(f"❌ Error: Training script not found: {train_script}")
        return False
    
    cmd = ["bash", train_script, scene_name, scene_dir, output_dir, export_dir]
    return run_command(cmd, "Semantic NeRF Training & Export")


def stage_3_voxelization(export_dir, scene_name, voxel_size=0.05):
    """Stage 3: Voxelize the point cloud."""
    print("\n" + "#" * 70)
    print("# STAGE 3: Voxelization")
    print("#" * 70)
    
    ply_path = os.path.join(export_dir, scene_name, "point_cloud.ply")
    npz_path = os.path.join(export_dir, scene_name, "voxel_grid.npz")
    
    if not os.path.exists(ply_path):
        print(f"❌ Error: Point cloud not found: {ply_path}")
        print("Please complete Stage 2 first.")
        return False
    
    if os.path.exists(npz_path):
        print(f"\n⚠️  Voxel grid already exists: {npz_path}")
        response = input("Regenerate voxel grid? (y/N): ")
        if response.lower() != 'y':
            print("Skipping voxelization")
            return True
    
    cmd = [
        "python", "voxelize/voxelize.py",
        ply_path, npz_path,
        "--voxel-size", str(voxel_size)
    ]
    
    return run_command(cmd, "Voxelization")


def stage_4_minecraft_conversion(export_dir, scene_name):
    """Stage 4: Convert voxel grid to Minecraft format."""
    print("\n" + "#" * 70)
    print("# STAGE 4: Minecraft Conversion")
    print("#" * 70)
    
    npz_path = os.path.join(export_dir, scene_name, "voxel_grid.npz")
    
    if not os.path.exists(npz_path):
        print(f"❌ Error: Voxel grid not found: {npz_path}")
        print("Please complete Stage 3 first.")
        return False
    
    # Check for conversion scripts
    convert_script = "export/convert.py"
    if not os.path.exists(convert_script):
        print(f"⚠️  Warning: Conversion script not found: {convert_script}")
        print("Skipping Minecraft conversion")
        return True
    
    cmd = ["python", convert_script, npz_path, export_dir, scene_name]
    return run_command(cmd, "Minecraft Conversion")


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Worldcraft Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene
  
Directory Structure:
  data/my_scene/
    images/          <- Input images
    semantics/       <- Generated semantic masks (Stage 1)
    panoptic_classes.json
  
  outputs/
    processed/       <- COLMAP processed data (Stage 2)
    models/          <- Trained NeRF models (Stage 2)
  
  exports/
    my_scene/
      point_cloud.ply     <- Exported point cloud (Stage 2)
      voxel_grid.npz      <- Voxelized grid (Stage 3)
      *.litematic          <- Minecraft file (Stage 4)
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input scene directory (must contain 'images/' subdirectory)"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for processed data and models (default: outputs)"
    )
    parser.add_argument(
        "--export",
        default="exports",
        help="Export directory for point clouds and final outputs (default: exports)"
    )
    parser.add_argument(
        "--scene-name",
        required=True,
        help="Scene name (used for organizing outputs)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size for voxelization (default: 0.05)"
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from specific stage (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    scene_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    export_dir = os.path.abspath(args.export)
    
    print("=" * 70)
    print("WORLDCRAFT PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"Scene directory: {scene_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Export directory: {export_dir}")
    print(f"Scene name: {args.scene_name}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Starting from stage: {args.start_stage}")
    
    # Validate input directory
    if not check_directory_structure(scene_dir):
        return 1
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    
    # Run pipeline stages
    success = True
    
    if args.start_stage <= 1:
        if not stage_1_semantic_segmentation(scene_dir):
            print("\n❌ Pipeline failed at Stage 1")
            return 1
    
    if args.start_stage <= 2:
        if not stage_2_nerf_training(scene_dir, output_dir, export_dir, args.scene_name):
            print("\n❌ Pipeline failed at Stage 2")
            return 1
    
    if args.start_stage <= 3:
        if not stage_3_voxelization(export_dir, args.scene_name, args.voxel_size):
            print("\n❌ Pipeline failed at Stage 3")
            return 1
    
    if args.start_stage <= 4:
        if not stage_4_minecraft_conversion(export_dir, args.scene_name):
            print("\n❌ Pipeline failed at Stage 4")
            return 1
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Semantics: {os.path.join(scene_dir, 'semantics')}")
    print(f"  - Models: {os.path.join(output_dir, 'models', args.scene_name)}")
    print(f"  - Point cloud: {os.path.join(export_dir, args.scene_name, 'point_cloud.ply')}")
    print(f"  - Voxel grid: {os.path.join(export_dir, args.scene_name, 'voxel_grid.npz')}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
