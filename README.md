# Worldcraft - Voxel to Minecraft Converter

Convert voxel data into Minecraft worlds that can be opened in Amulet Editor!

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Generate a Minecraft World
```bash
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_world \
    --format world
```

### Open in Amulet Editor
1. Download [Amulet Editor](https://www.amuletmc.com/)
2. Open Amulet and click "Select World"
3. Navigate to `outputs/my_world`
4. View your 3D structure!

## Features

- **World Export**: Creates Minecraft worlds compatible with Amulet Editor
- **mcfunction Export**: Generates command files for datapacks
- **Visualization**: Preview structures without Minecraft using matplotlib
- **Downscaling**: Reduce structure size for performance
- **Customizable**: Map semantic labels to any Minecraft blocks

## Available Scripts

| Script | Purpose |
|--------|---------|
| [export_from_vox.py](export_from_vox.py) | Main export tool (world + mcfunction) |
| [voxels_to_minecraft.py](voxels_to_minecraft.py) | Standalone Amulet world exporter |
| [visualize_structure.py](visualize_structure.py) | Preview structures with matplotlib |
| [create_test_data.py](create_test_data.py) | Generate sample voxel data |

## Documentation

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete documentation including:
- Detailed command-line arguments
- Input data format specifications
- Block mapping configuration
- Troubleshooting guide
- Example workflows

## Project Structure

```
test_data/sample_scene/     # Sample voxel data
  ├── occupancy.npy         # Binary occupancy grid
  ├── semantic_id.npy       # Semantic labels
  ├── rgb.npy              # RGB colors
  └── meta.json            # Metadata

block_maps/                 # Block mapping configs
  └── semantic_label_map.json

outputs/                    # Generated worlds/functions
```

## Example Output

The test data generates a simple structure with:
- 752 blocks total
- Stone brick floor
- Brick walls
- Oak plank roof
- Glass windows

## Requirements

- Python 3.7+
- numpy
- amulet-core
- amulet-nbt
- matplotlib (for visualization)

## Notes

- Supports Minecraft Java Edition 1.20.1
- Works with Amulet Editor for viewing/editing
- Can export to both world format and mcfunction commands
- Handles large structures efficiently

## Quick Test

```bash
# Run with test data
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/test_world \
    --format world \
    --downscale 1

# Check output
ls outputs/test_world/
# Should show: level.dat, region/, entities/, README.txt
```

## License

See project repository for license information.
