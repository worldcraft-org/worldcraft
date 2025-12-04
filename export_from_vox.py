import numpy as np
import json
import os
import argparse
from pathlib import Path

def load_voxel_data(vox_dir):
    """Load voxel data from directory"""
    print(f"Loading voxel data from {vox_dir}")
    
    occupancy = np.load(os.path.join(vox_dir, "occupancy.npy"))
    rgb = np.load(os.path.join(vox_dir, "rgb.npy"))
    semantic_id = np.load(os.path.join(vox_dir, "semantic_id.npy"))
    
    with open(os.path.join(vox_dir, "meta.json"), 'r') as f:
        meta = json.load(f)
    
    print(f"  Dimensions: {occupancy.shape}")
    print(f"  Occupied voxels: {np.sum(occupancy)}")
    
    return occupancy, rgb, semantic_id, meta

def downscale_voxels(occupancy, semantic_id, rgb, factor=2):
    """Downscale voxel grid by aggregation"""
    print(f"\nDownscaling by factor of {factor}")
    
    h, w, d = occupancy.shape
    new_h, new_w, new_d = h // factor, w // factor, d // factor
    
    # Initialize new arrays
    new_occupancy = np.zeros((new_h, new_w, new_d), dtype=np.uint8)
    new_semantic = np.zeros((new_h, new_w, new_d), dtype=np.uint8)
    new_rgb = np.zeros((new_h, new_w, new_d, 3), dtype=np.uint8)
    
    # Aggregate each block
    for i in range(new_h):
        for j in range(new_w):
            for k in range(new_d):
                # Extract the neighborhood
                i_start, i_end = i * factor, (i + 1) * factor
                j_start, j_end = j * factor, (j + 1) * factor
                k_start, k_end = k * factor, (k + 1) * factor
                
                occ_block = occupancy[i_start:i_end, j_start:j_end, k_start:k_end]
                sem_block = semantic_id[i_start:i_end, j_start:j_end, k_start:k_end]
                rgb_block = rgb[i_start:i_end, j_start:j_end, k_start:k_end]
                
                # Majority vote for occupancy
                new_occupancy[i, j, k] = 1 if np.sum(occ_block) > (factor**3 / 2) else 0
                
                # Mode for semantic_id (most common label)
                if new_occupancy[i, j, k]:
                    occupied_semantics = sem_block[occ_block > 0]
                    if len(occupied_semantics) > 0:
                        new_semantic[i, j, k] = np.bincount(occupied_semantics).argmax()
                
                # Mean for RGB
                if new_occupancy[i, j, k]:
                    new_rgb[i, j, k] = np.mean(rgb_block, axis=(0, 1, 2)).astype(np.uint8)
    
    print(f"  Original: {occupancy.shape}")
    print(f"  Downscaled: {new_occupancy.shape}")
    print(f"  Occupancy retained: {np.sum(new_occupancy)} / {np.sum(occupancy)} "
          f"({100 * np.sum(new_occupancy) / max(np.sum(occupancy), 1):.1f}%)")
    
    return new_occupancy, new_semantic, new_rgb

def load_block_map(label_map_path):
    """Load semantic label to Minecraft block mapping"""
    with open(label_map_path, 'r') as f:
        return json.load(f)

def print_label_histogram(semantic_id, meta, title="Label Distribution"):
    """Print distribution of semantic labels"""
    print(f"\n{title}:")
    unique, counts = np.unique(semantic_id, return_counts=True)
    labels = meta.get('labels', {})
    
    for label_id, count in zip(unique, counts):
        label_name = labels.get(str(label_id), f"unknown_{label_id}")
        print(f"  {label_name}: {count} voxels ({100*count/semantic_id.size:.1f}%)")

def export_to_mcfunction(occupancy, semantic_id, block_map, output_path, offset=(0, 0, 0)):
    """Export voxels to .mcfunction file (Minecraft commands)"""
    print(f"\nExporting to .mcfunction file: {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    mcfunction_file = os.path.join(output_path, "structure.mcfunction")
    
    h, w, d = occupancy.shape
    block_count = 0
    
    commands = []
    commands.append("# Generated structure from voxel data")
    commands.append(f"# Dimensions: {h}x{w}x{d}")
    commands.append("")
    
    # Place blocks
    for i in range(h):
        for j in range(w):
            for k in range(d):
                if occupancy[i, j, k]:
                    semantic_label = str(semantic_id[i, j, k])
                    
                    if semantic_label in block_map:
                        block_name = block_map[semantic_label]['block']
                        
                        # Calculate world position
                        x, y, z = offset[0] + i, offset[1] + j, offset[2] + k
                        
                        # Create setblock command
                        commands.append(f"setblock {x} {y} {z} {block_name}")
                        block_count += 1
    
    # Write to file
    with open(mcfunction_file, 'w') as f:
        f.write('\n'.join(commands))
    
    print(f"  ✓ Created {block_count} setblock commands")
    print(f"  ✓ File saved to: {mcfunction_file}")
    
    # Also create a summary file
    summary_file = os.path.join(output_path, "README.txt")
    with open(summary_file, 'w') as f:
        f.write("Minecraft Structure Export\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Structure size: {h}x{w}x{d}\n")
        f.write(f"Total blocks: {block_count}\n")
        f.write(f"Offset: ({offset[0]}, {offset[1]}, {offset[2]})\n\n")
        f.write("HOW TO USE:\n")
        f.write("-" * 50 + "\n")
        f.write("Method 1: Copy-paste commands\n")
        f.write("  1. Open structure.mcfunction\n")
        f.write("  2. Copy all commands\n")
        f.write("  3. In Minecraft, press T to open chat\n")
        f.write("  4. Paste and run commands (may need to do in batches)\n\n")
        f.write("Method 2: Use as datapack function\n")
        f.write("  1. Create a datapack folder structure\n")
        f.write("  2. Place structure.mcfunction in data/namespace/functions/\n")
        f.write("  3. Load datapack and run with /function namespace:structure\n\n")
        f.write("Note: Large structures may cause lag. Consider splitting into chunks.\n")
    
    print(f"  ✓ Instructions saved to: {summary_file}")
    
    return block_count

def export_to_amulet_world(occupancy, semantic_id, block_map, output_path, offset=(0, 64, 0)):
    """Export voxels to a Minecraft world that can be opened in Amulet Editor"""
    try:
        import amulet
        from amulet.api.block import Block
    except ImportError:
        print("ERROR: amulet-core not installed. Install with: pip install amulet-core amulet-nbt")
        return 0

    print(f"\nExporting to Minecraft world: {output_path}")

    # Create or load the world
    # First, try to create a new world if it doesn't exist
    if not os.path.exists(output_path):
        print("  Creating new Minecraft world...")
        # We need to create a minimal world structure
        # Amulet requires an existing world to load, so we'll use a workaround
        # by creating the directory structure manually
        os.makedirs(output_path, exist_ok=True)

        # Create level.dat file with basic world data
        import amulet_nbt

        # Create a minimal level.dat structure
        level_data = amulet_nbt.TAG_Compound({
            "Data": amulet_nbt.TAG_Compound({
                "version": amulet_nbt.TAG_Int(19133),  # Minecraft 1.20.1
                "LevelName": amulet_nbt.TAG_String(os.path.basename(output_path)),
                "GameType": amulet_nbt.TAG_Int(1),  # Creative mode
                "Difficulty": amulet_nbt.TAG_Byte(0),
                "DifficultyLocked": amulet_nbt.TAG_Byte(0),
                "Time": amulet_nbt.TAG_Long(0),
                "DayTime": amulet_nbt.TAG_Long(0),
                "SpawnX": amulet_nbt.TAG_Int(offset[0]),
                "SpawnY": amulet_nbt.TAG_Int(offset[1]),
                "SpawnZ": amulet_nbt.TAG_Int(offset[2]),
                "allowCommands": amulet_nbt.TAG_Byte(1),
                "clearWeatherTime": amulet_nbt.TAG_Int(0),
                "rainTime": amulet_nbt.TAG_Int(0),
                "raining": amulet_nbt.TAG_Byte(0),
                "thunderTime": amulet_nbt.TAG_Int(0),
                "thundering": amulet_nbt.TAG_Byte(0),
                "Version": amulet_nbt.TAG_Compound({
                    "Id": amulet_nbt.TAG_Int(3218),
                    "Name": amulet_nbt.TAG_String("1.20.1"),
                    "Snapshot": amulet_nbt.TAG_Byte(0)
                }),
                "DataVersion": amulet_nbt.TAG_Int(3465)
            })
        })

        # Save level.dat
        level_dat_path = os.path.join(output_path, "level.dat")
        level_data.save_to(level_dat_path, compressed=True)
        print("  [OK] Created level.dat")

    # Now load the world
    level = amulet.load_level(output_path)

    h, w, d = occupancy.shape
    block_count = 0

    print(f"  Dimensions: {h}x{w}x{d}")
    print(f"  Placing blocks...")

    # Place blocks
    for i in range(h):
        if i % 10 == 0:
            print(f"    Progress: {i}/{h} ({100*i/h:.1f}%)")

        for j in range(w):
            for k in range(d):
                if occupancy[i, j, k]:
                    semantic_label = str(semantic_id[i, j, k])

                    if semantic_label in block_map:
                        block_name = block_map[semantic_label]['block']

                        # Parse block name
                        if ':' in block_name:
                            namespace, base_name = block_name.split(':', 1)
                        else:
                            namespace = "minecraft"
                            base_name = block_name

                        # Calculate world position
                        x, y, z = offset[0] + i, offset[1] + j, offset[2] + k

                        # Create and place the block
                        block = Block(namespace, base_name)

                        try:
                            level.set_version_block(
                                x, y, z,
                                "minecraft:overworld",
                                ("java", (1, 20, 1)),
                                block
                            )
                            block_count += 1
                        except Exception as e:
                            if block_count == 0:  # Only print first error
                                print(f"    Warning: Error placing block: {e}")

    print(f"  [OK] Placed {block_count} blocks")
    print(f"  Saving world...")
    level.save()
    level.close()
    print(f"  [OK] World saved!")

    # Create instructions
    readme_path = os.path.join(output_path, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Minecraft World - Generated from Voxel Data\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Structure size: {h}x{w}x{d}\n")
        f.write(f"Total blocks: {block_count}\n")
        f.write(f"Offset: ({offset[0]}, {offset[1]}, {offset[2]})\n\n")
        f.write("TO OPEN IN AMULET EDITOR:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Download Amulet Editor: https://www.amuletmc.com/\n")
        f.write("2. Open Amulet and click 'Select World'\n")
        f.write(f"3. Navigate to this folder: {os.path.abspath(output_path)}\n")
        f.write("4. View and edit your structure!\n")

    print(f"  [OK] Instructions saved to: {readme_path}")

    return block_count


def main():
    parser = argparse.ArgumentParser(description='Export voxel data to Minecraft')
    parser.add_argument('--vox_dir', required=True, help='Path to voxel data directory')
    parser.add_argument('--downscale', type=int, default=2, help='Downscale factor (default: 2)')
    parser.add_argument('--label_map', required=True, help='Path to semantic label map JSON')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--offset_x', type=int, default=0, help='X offset in world')
    parser.add_argument('--offset_y', type=int, default=100, help='Y offset in world')
    parser.add_argument('--offset_z', type=int, default=0, help='Z offset in world')
    parser.add_argument('--format', choices=['mcfunction', 'world', 'both'], default='both',
                       help='Output format: mcfunction (commands), world (Amulet), or both')

    args = parser.parse_args()

    print("="*60)
    print("Minecraft Voxel Export Pipeline")
    print("="*60)

    # Load data
    occupancy, rgb, semantic_id, meta = load_voxel_data(args.vox_dir)

    # Print original histogram
    print_label_histogram(semantic_id, meta, "Original Label Distribution")

    # Downscale
    occupancy_down, semantic_down, rgb_down = downscale_voxels(
        occupancy, semantic_id, rgb, factor=args.downscale
    )

    # Print downscaled histogram
    print_label_histogram(semantic_down, meta, "Downscaled Label Distribution")

    # Load block mapping
    block_map = load_block_map(args.label_map)
    print(f"\nLoaded {len(block_map)} block mappings")

    block_count = 0

    # Export to mcfunction
    if args.format in ['mcfunction', 'both']:
        mcfunction_out = args.out if args.format == 'mcfunction' else args.out + '_mcfunction'
        block_count = export_to_mcfunction(
            occupancy_down,
            semantic_down,
            block_map,
            mcfunction_out,
            offset=(args.offset_x, args.offset_y, args.offset_z)
        )

    # Export to Amulet world
    if args.format in ['world', 'both']:
        world_out = args.out if args.format == 'world' else args.out + '_world'
        block_count = export_to_amulet_world(
            occupancy_down,
            semantic_down,
            block_map,
            world_out,
            offset=(args.offset_x, args.offset_y, args.offset_z)
        )

    print("\n" + "="*60)
    print("[SUCCESS] Export Complete!")
    print("="*60)
    print(f"\nGenerated structure with {block_count} blocks")
    if args.format in ['world', 'both']:
        world_out = args.out if args.format == 'world' else args.out + '_world'
        print(f"\nWorld location: {os.path.abspath(world_out)}")
        print("Open with Amulet Editor!")
    if args.format in ['mcfunction', 'both']:
        mcfunction_out = args.out if args.format == 'mcfunction' else args.out + '_mcfunction'
        print(f"\nMcfunction location: {os.path.abspath(mcfunction_out)}")
    print("\nSee README.txt files for usage instructions.")

if __name__ == "__main__":
    main()