# Worldcraft - Voxel to Minecraft Converter

Complete pipeline for converting voxel data into Minecraft worlds that can be opened in Amulet Editor.

## Installation

1. **Install required packages:**
```bash
pip install numpy amulet-core amulet-nbt
```

## Quick Start

### Option 1: Create Minecraft World (Recommended for Amulet Editor)

```bash
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_world \
    --format world \
    --downscale 1
```

This creates a Minecraft world that can be opened directly in Amulet Editor!

### Option 2: Generate mcfunction Commands

```bash
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_structure \
    --format mcfunction \
    --downscale 2
```

This generates `.mcfunction` files with setblock commands.

### Option 3: Generate Both Formats

```bash
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_export \
    --format both \
    --downscale 2
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vox_dir` | Yes | - | Path to directory containing voxel data (occupancy.npy, rgb.npy, semantic_id.npy, meta.json) |
| `--label_map` | Yes | - | Path to JSON file mapping semantic labels to Minecraft blocks |
| `--out` | Yes | - | Output directory path |
| `--format` | No | `both` | Output format: `world`, `mcfunction`, or `both` |
| `--downscale` | No | `2` | Downscale factor (1 = no downscaling, 2 = half size, etc.) |
| `--offset_x` | No | `0` | X offset in world coordinates |
| `--offset_y` | No | `100` | Y offset in world coordinates |
| `--offset_z` | No | `0` | Z offset in world coordinates |

## Input Data Format

Your voxel directory must contain:

1. **occupancy.npy** - Binary 3D array indicating occupied voxels (shape: H×W×D)
2. **semantic_id.npy** - Integer 3D array with semantic labels (shape: H×W×D)
3. **rgb.npy** - RGB color values (shape: H×W×D×3)
4. **meta.json** - Metadata including:
   ```json
   {
     "dimensions": [16, 16, 16],
     "labels": {
       "0": "air",
       "1": "floor",
       "2": "wall"
     }
   }
   ```

## Block Mapping Format

The label map JSON maps semantic IDs to Minecraft blocks:

```json
{
  "0": {
    "block": "minecraft:air",
    "description": "Empty space"
  },
  "1": {
    "block": "minecraft:stone_bricks",
    "description": "Floor"
  },
  "2": {
    "block": "minecraft:bricks",
    "description": "Wall"
  }
}
```

### Common Minecraft Block Names

- `minecraft:stone_bricks`
- `minecraft:bricks`
- `minecraft:oak_planks`
- `minecraft:glass`
- `minecraft:concrete` (colors: `white_concrete`, `gray_concrete`, etc.)
- `minecraft:wool` (colors: `white_wool`, `red_wool`, etc.)
- `minecraft:terracotta` (colors: `white_terracotta`, etc.)

See [Minecraft Wiki](https://minecraft.fandom.com/wiki/Java_Edition_data_values) for complete block list.

## Opening in Amulet Editor

1. **Download Amulet Editor:**
   - Visit: https://www.amuletmc.com/
   - Download and install for your platform

2. **Open your world:**
   - Launch Amulet Editor
   - Click "Select World"
   - Navigate to your output directory (e.g., `outputs/my_world`)
   - Select the world and click "Open"

3. **View and edit:**
   - Use mouse to navigate (WASD for movement, mouse for camera)
   - View your voxel structure converted to Minecraft blocks
   - Edit, export, or convert to other formats

## Alternative: Using mcfunction Files

If you generated `.mcfunction` files:

### Method 1: Direct Command Execution (Small structures)

1. Open Minecraft (Java Edition)
2. Create a creative world
3. Open the `.mcfunction` file in a text editor
4. Copy commands in batches
5. Paste into Minecraft chat (press T)

### Method 2: As a Datapack (Recommended for large structures)

1. Create datapack structure:
   ```
   my_datapack/
   ├── pack.mcmeta
   └── data/
       └── mystructure/
           └── functions/
               └── structure.mcfunction
   ```

2. Create `pack.mcmeta`:
   ```json
   {
     "pack": {
       "pack_format": 15,
       "description": "Generated structure"
     }
   }
   ```

3. Copy generated `.mcfunction` file to `functions/` folder

4. Install datapack:
   - Copy `my_datapack` to `.minecraft/saves/[world_name]/datapacks/`
   - In game: `/reload`
   - Run: `/function mystructure:structure`

## Visualization (Without Minecraft)

Preview your structure using matplotlib:

```bash
python visualize_structure.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --view both \
    --save preview.png
```

Options:
- `--view 3d` - Interactive 3D view
- `--view 2d` - Top/front/side views + statistics
- `--view both` - Both visualizations
- `--save` - Save to file instead of displaying

## Performance Tips

### For Large Structures

1. **Use downscaling:**
   ```bash
   --downscale 4  # Reduces to 1/4 size in each dimension
   ```

2. **Process in chunks:**
   - Split large voxel grids into smaller regions
   - Process each region separately
   - Use different offsets for each chunk

3. **Use world format over mcfunction:**
   - World format is much more efficient
   - Can handle millions of blocks
   - mcfunction has command limits

### Memory Optimization

- Large voxel grids may require significant RAM
- Consider processing subregions if you encounter memory errors
- Downscaling reduces memory requirements exponentially

## Troubleshooting

### Issue: "amulet-core not installed"
**Solution:** Run `pip install amulet-core amulet-nbt`

### Issue: "Could not find a matching format"
**Solution:** The world creation failed. Check permissions and ensure output directory is writable.

### Issue: Blocks appear as air in Amulet
**Solution:** Check that block names in label_map.json match Minecraft block IDs exactly (use `minecraft:` prefix)

### Issue: Structure appears at wrong location
**Solution:** Adjust `--offset_x`, `--offset_y`, `--offset_z` parameters

### Issue: Too many blocks (lag in Minecraft)
**Solution:** Increase `--downscale` factor or split into multiple regions

## Example Workflows

### Complete Pipeline from Scratch

```bash
# 1. Generate test data (if needed)
python create_test_data.py

# 2. Visualize first
python visualize_structure.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --view both

# 3. Export to Minecraft world
python export_from_vox.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_world \
    --format world \
    --downscale 1

# 4. Open in Amulet Editor
# (Launch Amulet and select outputs/my_world)
```

### Custom Voxel Data

```python
import numpy as np
import json

# Create your voxel data
occupancy = np.zeros((32, 32, 32), dtype=np.uint8)
semantic_id = np.zeros((32, 32, 32), dtype=np.uint8)
rgb = np.zeros((32, 32, 32, 3), dtype=np.uint8)

# Example: Create a simple cube
occupancy[10:20, 10:20, 10:20] = 1
semantic_id[10:20, 10:20, 10:20] = 1
rgb[10:20, 10:20, 10:20] = [255, 0, 0]  # Red

# Save
output_dir = "my_voxels"
np.save(f"{output_dir}/occupancy.npy", occupancy)
np.save(f"{output_dir}/semantic_id.npy", semantic_id)
np.save(f"{output_dir}/rgb.npy", rgb)

# Save metadata
meta = {
    "dimensions": list(occupancy.shape),
    "labels": {"0": "air", "1": "block"}
}
with open(f"{output_dir}/meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

# Create label map
label_map = {
    "0": {"block": "minecraft:air", "description": "Empty"},
    "1": {"block": "minecraft:red_concrete", "description": "Cube"}
}
with open("my_label_map.json", 'w') as f:
    json.dump(label_map, f, indent=2)

# Now run export!
```

## Project Structure

```
Worldcraft/
├── export_from_vox.py          # Main export script (both formats)
├── voxels_to_minecraft.py      # Standalone Amulet exporter
├── visualize_structure.py      # Preview tool
├── create_test_data.py         # Generate test voxels
├── test_amulet.py             # Test Amulet installation
├── block_maps/                 # Block mapping JSONs
│   └── semantic_label_map.json
├── test_data/                  # Sample voxel data
│   └── sample_scene/
└── outputs/                    # Generated worlds/functions
```

## Additional Scripts

### voxels_to_minecraft.py

Standalone script for creating Minecraft worlds (same as `--format world`):

```bash
python voxels_to_minecraft.py \
    --vox_dir test_data/sample_scene \
    --label_map block_maps/semantic_label_map.json \
    --out outputs/my_world \
    --downscale 1
```

## Support

- **Amulet Documentation:** https://github.com/Amulet-Team/Amulet-Core
- **Minecraft Block IDs:** https://minecraft.fandom.com/wiki/Java_Edition_data_values
- **PyMCTranslate:** https://github.com/gentlegiantJGC/PyMCTranslate

## Notes

- The pipeline supports Minecraft Java Edition 1.20.1
- World generation may show warnings about WorldGenSettings - these are safe to ignore
- For production use, test with small structures first
- Amulet Editor can also convert between Java/Bedrock editions
