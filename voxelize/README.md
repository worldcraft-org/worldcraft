# NeRF to Voxel Grid Converter

Converts NeRF scene representations into dense semantic voxel grids for Minecraft export. Supports multiple NeRF dataset formats including synthetic, LLFF, real_360, and Replica datasets.

## Installation

### Requirements
- Python 3.8+
- NumPy
- Matplotlib
- PyTorch (optional, for advanced NeRF models)
- PIL/Pillow (optional, for image loading)
- OpenCV (optional, for enhanced visualizations)

### Setup
```bash
# Clone the repository (or create project directory)
mkdir worldcraft_voxel
cd worldcraft_voxel

# Install dependencies
pip install -r requirements.txt

# Copy nerf_to_voxel.py to this directory
```

## Usage

### Basic Usage (with dummy NeRF)
```bash
python nerf_to_voxel.py --use_dummy
```

### Real NeRF Dataset Usage
```bash
# For synthetic NeRF datasets
python nerf_to_voxel.py --dataset_type synthetic --scene lego --archive_dir /path/to/datasets

# For LLFF datasets
python nerf_to_voxel.py --dataset_type llff --scene fern --archive_dir /path/to/datasets

# For Replica datasets
python nerf_to_voxel.py --dataset_type replica --scene apartment_0 --replica_dir ./Replica-Dataset

# Direct dataset path
python nerf_to_voxel.py --dataset /path/to/specific/dataset/folder
```

### Custom Parameters
```bash
python nerf_to_voxel.py \
  --voxel_size 0.20 \
  --bbox -5 5 -5 5 0 15 \
  --out outputs/test_scene
```

### Arguments
- `--voxel_size`: Size of each voxel in meters (default: 0.15)
- `--bbox`: Bounding box as 6 floats: xmin xmax ymin ymax zmin zmax (default: -10 10 -10 10 0 20)
- `--out`: Output directory path (default: outputs/scene_001)
- `--dataset`: Direct path to NeRF dataset directory
- `--dataset_type`: Type of dataset (synthetic, llff, real_360, replica)
- `--scene`: Scene name within the dataset type
- `--use_dummy`: Use dummy NeRF model instead of real dataset
- `--density_threshold`: Density threshold for occupancy determination (default: 0.5)
- `--batch_size`: Batch size for processing to avoid OOM (default: 10000)
- `--archive_dir`: Directory containing NeRF datasets (default: ./archive)
- `--replica_dir`: Directory containing Replica datasets (default: ./Replica-Dataset)

## Output Format

The script generates a structured directory containing:

```
outputs/scene_001/
├── occupancy.npy          # Boolean array (X, Y, Z) - True if voxel is solid
├── rgb.npy                # uint8 array (X, Y, Z, 3) - RGB color [0-255]
├── semantic_id.npy        # int32 array (X, Y, Z) - Semantic class ID
├── meta.json              # Metadata (grid spec, transforms, labels)
└── slices_visualization.png  # Quick visual check (XY, XZ, YZ slices)
```

### Loading Outputs

```python
import numpy as np
import json

# Load voxel data
occupancy = np.load('outputs/scene_001/occupancy.npy')
rgb = np.load('outputs/scene_001/rgb.npy')
semantic_id = np.load('outputs/scene_001/semantic_id.npy')

# Load metadata
with open('outputs/scene_001/meta.json') as f:
    meta = json.load(f)

print(f"Grid size: {meta['grid_size']}")
print(f"Voxel size: {meta['voxel_size_m']}m")
```

## Coordinate System

- **Origin**: Bounding box minimum corner
- **Axes**: ENU (East-North-Up) convention
  - X-axis: East (right)
  - Y-axis: North (forward)
  - Z-axis: Up
- **Handedness**: Right-handed coordinate system
- **Units**: Meters

### Coordinate Transformations

World to Voxel Index:
```
voxel_index = floor((world_coord - bbox_min) / voxel_size)
```

Voxel Index to World (center):
```
world_coord = bbox_min + (voxel_index + 0.5) * voxel_size
```

## Semantic Labels (Week 1)

Current label set (placeholder for testing):
- `0`: air/void
- `1`: building
- `2`: vegetation

This will be updated when real Semantic-NeRF data is available.

## Supported Dataset Types

1. **Synthetic NeRF**: Standard synthetic datasets with `transforms_train.json`
2. **LLFF**: Real forward-facing scenes with `poses_bounds.npy`
3. **Real 360**: 360-degree real scenes
4. **Replica**: Facebook AI Research Replica dataset (simulated indoor scenes)
5. **Dummy NeRF**: Synthetic test data for development

## Features

- ✅ Multiple NeRF dataset format support
- ✅ Automatic dataset type detection
- ✅ Chunked processing for large scenes
- ✅ Configurable density thresholds
- ✅ Batch processing to avoid OOM
- ✅ Optional dependencies with graceful fallbacks
- ✅ Comprehensive error handling

## Future Enhancements

- [ ] Support for more semantic classes
- [ ] Integration with trained Semantic-NeRF models
- [ ] GPU acceleration for faster processing
- [ ] Adaptive voxel sizing based on scene complexity

## Integration with Export Team

The Export team can use the outputs as follows:

```python
# Example: Read voxel grid and convert to Minecraft
import numpy as np

occupancy = np.load('outputs/scene_001/occupancy.npy')
rgb = np.load('outputs/scene_001/rgb.npy')
semantic_id = np.load('outputs/scene_001/semantic_id.npy')

# Iterate through occupied voxels
for x in range(occupancy.shape[0]):
    for y in range(occupancy.shape[1]):
        for z in range(occupancy.shape[2]):
            if occupancy[x, y, z]:
                color = rgb[x, y, z]  # RGB values
                block_type = semantic_id[x, y, z]  # Semantic class
                # Convert to Minecraft block...
```

## Troubleshooting

**Memory Error**: Reduce voxel size or bounding box dimensions
```bash
python nerf_to_voxel.py --voxel_size 0.20 --bbox -5 5 -5 5 0 10
```

**Import Error**: Install dependencies
```bash
pip install numpy matplotlib
# Optional dependencies
pip install torch torchvision  # For PyTorch support
pip install pillow opencv-python  # For image processing
```

**Dataset Not Found**: Check dataset paths and structure
```bash
# Verify dataset directory exists and contains required files
ls -la /path/to/dataset/
# For synthetic: look for transforms_train.json
# For LLFF: look for poses_bounds.npy
# For Replica: look for .ply files
```

## Contact

For questions or issues, contact the Voxelization subteam during weekly meetings.

## Version

- **Version**: v0.2
- **Date**: Updated with real NeRF dataset support
- **Status**: Production ready for multiple NeRF dataset formats