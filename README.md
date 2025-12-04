# Worldcraft Pipeline

A complete pipeline for converting real-world scenes into Minecraft worlds using Neural Radiance Fields (NeRF) and semantic segmentation.

## Overview

The Worldcraft pipeline transforms images of real-world scenes into voxelized 3D representations suitable for Minecraft. The pipeline consists of four main stages:

1. **Semantic Segmentation** - Process images to generate semantic masks using Mask2Former
2. **NeRF Training** - Train a semantic NeRF model and export a point cloud
3. **Voxelization** - Convert the point cloud into a voxel grid
4. **Minecraft Conversion** - Transform voxels into Minecraft blocks

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (required for NeRF training)
- Recommended: 16GB+ RAM, 50GB+ free disk space

### Software Requirements
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- NVIDIA GPU drivers
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/worldcraft-org/worldcraft.git
cd worldcraft
```

### 2. Run Setup Script

The setup script will:
- Verify conda and GPU availability
- Create the conda environment with all dependencies
- Install CUDA toolkit 11.8
- Pre-download required models (~1.5GB)
- Create necessary directories

```bash
bash setup.sh
```

This process may take 15-30 minutes depending on your internet connection.

### 3. Activate the Environment

```bash
conda activate worldcraft
```

## Quick Start

### Prepare Your Data

1. Create a scene directory with your images:

```bash
mkdir -p data/my_scene/images
# Copy your images to data/my_scene/images/
```

Images should be:
- JPG or PNG format
- Captured from multiple viewpoints
- Overlapping coverage of the scene
- 20-100 images recommended

### Run the Complete Pipeline

```bash
python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene
```

This will execute all four stages automatically. The pipeline will:
- Generate semantic segmentation masks
- Train a NeRF model (this takes the longest, 2-8 hours)
- Export and voxelize the point cloud
- Convert to Minecraft format

### Output Structure

```
data/my_scene/
  images/              # Your input images
  semantics/           # Generated semantic masks (Stage 1)
  panoptic_classes.json

outputs/
  processed/my_scene/  # COLMAP processed data (Stage 2)
  models/my_scene/     # Trained NeRF models (Stage 2)

exports/
  my_scene/
    point_cloud.ply    # Exported point cloud (Stage 2)
    voxel_grid.npz     # Voxelized grid (Stage 3)
    *.litematic        # Minecraft file (Stage 4)
```

## Detailed Usage

### Stage 1: Semantic Segmentation

Process images to generate semantic segmentation masks:

```bash
python image-processing/process_semantics.py --scene-dir ./data/my_scene
```

**Options:**
- `--scene-dir`: Path to scene directory containing `images/` subdirectory

**Output:**
- `semantics/` - Full-resolution semantic masks
- `semantics_2/`, `semantics_4/`, `semantics_8/` - Downscaled versions
- `panoptic_classes.json` - Class labels and colors

**Time:** ~5-15 minutes for 50 images

### Stage 2: NeRF Training & Point Cloud Export

Train a semantic NeRF model and export a point cloud:

```bash
bash semnerf/train_job.sh my_scene ./data/my_scene ./outputs ./exports
```

**Parameters:**
1. `SCENE_NAME` - Name for organizing outputs
2. `DATA_DIR` - Path to scene directory (with images/ and semantics/)
3. `OUTPUT_DIR` - Path for processed data and models
4. `EXPORT_DIR` - Path for exported point clouds

**Steps:**
1. Process data with COLMAP (structure-from-motion)
2. Copy semantic annotations to processed data
3. Train semantic-nerfw model
4. Export point cloud with 1M points

**Time:** 2-8 hours depending on scene complexity and GPU

**Customization:**

To adjust training parameters, edit the `ns-train` command in `semnerf/train_job.sh`:

```bash
ns-train semantic-nerfw \
    --data "$OUTPUT_DIR/processed/$SCENE_NAME" \
    --output-dir "$OUTPUT_DIR/models/$SCENE_NAME" \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.pixel-sampler-num-rays-per-batch 4096 \
    --max-num-iterations 30000  # Add this to train longer
```

### Stage 3: Voxelization

Convert the point cloud into a voxel grid:

```bash
python voxelize/voxelize.py exports/my_scene/point_cloud.ply exports/my_scene/voxel_grid.npz --voxel-size 0.05
```

**Options:**
- `--voxel-size`: Size of each voxel in meters (default: 0.05)
  - Smaller = more detail but larger file size
  - Recommended range: 0.03-0.1

**Alternative - Web UI:**

```bash
python voxelize/voxelize.py serve
# Open http://localhost:8000/docs in your browser
```

**Time:** ~1-5 minutes

### Stage 4: Minecraft Conversion

Convert the voxel grid to Minecraft format:

```bash
python export/convert.py exports/my_scene/voxel_grid.npz exports/my_scene my_scene
```

**Output:** `.litematic` file that can be imported into Minecraft using Litematica mod

## Advanced Usage

### Running Individual Stages

You can run the pipeline from any stage using the `--start-stage` option:

```bash
# Skip semantic segmentation, start from NeRF training
python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene --start-stage 2

# Only run voxelization and conversion (stages 3-4)
python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene --start-stage 3
```

### Custom Voxel Sizes

```bash
# Higher detail (smaller voxels)
python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene --voxel-size 0.03

# Lower detail (larger voxels, faster processing)
python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene --voxel-size 0.1
```

### Processing Multiple Scenes

```bash
# Scene 1
python orchestrate.py --input data/scene1 --output outputs --scene-name scene1

# Scene 2
python orchestrate.py --input data/scene2 --output outputs --scene-name scene2
```

Each scene maintains separate directories in `outputs/` and `exports/`.

## Troubleshooting

### GPU Not Detected

If you see "CUDA not available":

1. Verify GPU drivers: `nvidia-smi`
2. Check PyTorch CUDA:
   ```bash
   conda activate worldcraft
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. If false, reinstall PyTorch with CUDA:
   ```bash
   conda activate worldcraft
   pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory Errors

If training fails with OOM errors:

1. Reduce batch size in `semnerf/train_job.sh`:
   ```bash
   --pipeline.datamanager.pixel-sampler-num-rays-per-batch 2048
   ```

2. Reduce image resolution by downscaling input images

3. Use fewer images (40-60 is usually sufficient)

### COLMAP Fails to Process Images

If structure-from-motion fails:

1. Ensure images have sufficient overlap
2. Check that images are clear and well-lit
3. Try reducing the number of images
4. Ensure images are from different viewpoints

### Semantic Segmentation Issues

If semantic masks look incorrect:

1. Check input image quality
2. Verify the model downloaded correctly:
   ```bash
   python scripts/download_models.py
   ```
3. The model is trained on urban scenes - performance may vary for other environments

### Point Cloud Export Issues

If point cloud is empty or sparse:

1. Check that NeRF training completed successfully
2. Review training logs in `outputs/models/your_scene/`
3. Verify semantic annotations exist in processed data
4. Try training longer (increase `--max-num-iterations`)

## Directory Structure

```
worldcraft/
├── data/                          # Input data
│   └── [scene_name]/
│       ├── images/                # Input images (you provide)
│       ├── semantics/             # Generated semantic masks
│       └── panoptic_classes.json  # Class definitions
│
├── outputs/                       # Processing outputs
│   ├── processed/                 # COLMAP processed data
│   └── models/                    # Trained NeRF models
│
├── exports/                       # Final exports
│   └── [scene_name]/
│       ├── point_cloud.ply
│       ├── voxel_grid.npz
│       └── *.litematic
│
├── image-processing/              # Semantic segmentation
│   └── process_semantics.py
│
├── semnerf/                       # NeRF training
│   └── train_job.sh
│
├── voxelize/                      # Voxelization
│   └── voxelize.py
│
├── export/                        # Minecraft conversion
│   └── convert.py
│
├── scripts/                       # Utility scripts
│   └── download_models.py
│
├── orchestrate.py                 # Main pipeline orchestrator
├── setup.sh                       # Setup script
├── environment.yml                # Conda environment
└── README.md                      # This file
```

## Technical Details

### Dependencies

Key dependencies installed via `environment.yml`:
- **Python 3.10**
- **PyTorch 2.1.2** with CUDA 11.8
- **Nerfstudio** - NeRF training framework
- **tiny-cuda-nn** - Fast neural network library
- **Transformers** - Hugging Face library for Mask2Former
- **Open3D** - Point cloud processing
- **FastAPI** - Voxelization web API

### Models

The pipeline uses:
- **Mask2Former** (facebook/mask2former-swin-large-mapillary-vistas-semantic)
  - Pre-trained on Mapillary Vistas dataset
  - 65 semantic classes
  - ~1.5GB model size

### NPZ Format

The voxel grid NPZ file contains:
- `points`: (N, 3) array of voxel center positions
- `color_grid`: (N, 3) array of RGB colors (uint8)
- `occupancy_grid`: (X, Y, Z) boolean array of voxel occupancy

## SLURM/HPC Usage

The pipeline can also run on SLURM clusters. The `semnerf/train_job.sh` script includes SLURM directives that are automatically ignored when running standalone.

To submit as a SLURM job:

```bash
sbatch semnerf/train_job.sh my_scene ./data/my_scene ./outputs ./exports
```

See `semnerf/README.md` for HPC-specific documentation.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{worldcraft2024,
  title = {Worldcraft: Real-World to Minecraft Pipeline},
  author = {Worldcraft Contributors},
  year = {2024},
  url = {https://github.com/worldcraft-org/worldcraft}
}
```

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

- [Nerfstudio](https://docs.nerf.studio/) - NeRF training framework
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) - Semantic segmentation
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) - Fast CUDA neural networks

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/worldcraft-org/worldcraft/issues)
- Check existing issues for solutions
- Provide error messages and logs when reporting problems

## Version History

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
