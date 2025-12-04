"""
Convert voxel data to a Minecraft world that can be opened in Amulet Editor
"""
import numpy as np
import json
import os
import argparse
from pathlib import Path
import amulet
from amulet.api.level import World
from amulet.api.block import Block
from amulet.api.errors import ChunkDoesNotExist


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


def load_block_map(label_map_path):
    """Load semantic label to Minecraft block mapping"""
    with open(label_map_path, 'r') as f:
        return json.load(f)


def create_minecraft_world(occupancy, semantic_id, block_map, output_path, offset=(0, 64, 0)):
    """
    Create a Minecraft world file that can be opened in Amulet Editor

    Args:
        occupancy: numpy array of occupied voxels
        semantic_id: numpy array of semantic labels
        block_map: mapping from semantic labels to Minecraft blocks
        output_path: path to save the world
        offset: (x, y, z) offset in the world coordinates
    """
    print(f"\nCreating Minecraft world at: {output_path}")

    # Create the world
    # Use Java Edition format (can be changed to Bedrock if needed)
    level = amulet.load_level(output_path)

    h, w, d = occupancy.shape
    block_count = 0
    air_block = Block("minecraft", "air")

    print(f"  Dimensions: {h}x{w}x{d}")
    print(f"  Placing blocks...")

    # Track which chunks we need to save
    modified_chunks = set()

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

                        # Parse block name (format: "minecraft:block_name")
                        if ':' in block_name:
                            namespace, base_name = block_name.split(':', 1)
                        else:
                            namespace = "minecraft"
                            base_name = block_name

                        # Calculate world position
                        x, y, z = offset[0] + i, offset[1] + j, offset[2] + k

                        # Create the block
                        block = Block(namespace, base_name)

                        # Set the block in the world
                        try:
                            level.set_version_block(
                                x, y, z,
                                "minecraft:overworld",  # dimension
                                ("java", (1, 20, 1)),   # version tuple
                                block
                            )
                            block_count += 1

                            # Track modified chunk
                            chunk_x, chunk_z = x >> 4, z >> 4
                            modified_chunks.add((chunk_x, chunk_z))
                        except Exception as e:
                            print(f"    Warning: Failed to place block at ({x}, {y}, {z}): {e}")

    print(f"  ✓ Placed {block_count} blocks")
    print(f"  Modified chunks: {len(modified_chunks)}")

    # Save the world
    print(f"  Saving world...")
    level.save()
    level.close()

    print(f"  ✓ World saved successfully!")

    # Create instructions file
    instructions_path = os.path.join(output_path, "README.txt")
    with open(instructions_path, 'w') as f:
        f.write("Minecraft World Generated from Voxel Data\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Structure size: {h}x{w}x{d}\n")
        f.write(f"Total blocks placed: {block_count}\n")
        f.write(f"World offset: ({offset[0]}, {offset[1]}, {offset[2]})\n\n")
        f.write("HOW TO OPEN:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Open Amulet Editor (https://www.amuletmc.com/)\n")
        f.write("2. Click 'Select World'\n")
        f.write(f"3. Navigate to: {os.path.abspath(output_path)}\n")
        f.write("4. Open and view your structure!\n\n")
        f.write("ALTERNATIVE - Open in Minecraft:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Copy this world folder to your Minecraft saves directory:\n")
        f.write("   Windows: %APPDATA%\\.minecraft\\saves\\\n")
        f.write("   Mac: ~/Library/Application Support/minecraft/saves/\n")
        f.write("   Linux: ~/.minecraft/saves/\n")
        f.write("2. Launch Minecraft and select the world\n\n")

    print(f"  ✓ Instructions saved to: {instructions_path}")

    return block_count


def create_empty_world(output_path):
    """Create an empty Minecraft world template"""
    print(f"Creating empty world template at: {output_path}")

    # Make sure parent directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create world using Amulet's creation method
    try:
        # Try to create a new world
        level = amulet.load_level(output_path)
        print("  ✓ World initialized")
        return level
    except Exception as e:
        print(f"  Note: {e}")
        # If world doesn't exist, we need to create it differently
        # This is handled by amulet internally when we start setting blocks
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert voxel data to Minecraft world for Amulet Editor')
    parser.add_argument('--vox_dir', required=True, help='Path to voxel data directory')
    parser.add_argument('--label_map', required=True, help='Path to semantic label map JSON')
    parser.add_argument('--out', required=True, help='Output world directory (will be created)')
    parser.add_argument('--offset_x', type=int, default=0, help='X offset in world')
    parser.add_argument('--offset_y', type=int, default=64, help='Y offset in world (default: 64)')
    parser.add_argument('--offset_z', type=int, default=0, help='Z offset in world')
    parser.add_argument('--downscale', type=int, default=1, help='Downscale factor (default: 1, no downscaling)')

    args = parser.parse_args()

    print("=" * 70)
    print("Voxel to Minecraft World Converter (Amulet-Compatible)")
    print("=" * 70)

    # Load data
    occupancy, rgb, semantic_id, meta = load_voxel_data(args.vox_dir)

    # Downscale if requested
    if args.downscale > 1:
        print(f"\nDownscaling by factor of {args.downscale}...")
        from export_from_vox import downscale_voxels
        occupancy, semantic_id, rgb = downscale_voxels(
            occupancy, semantic_id, rgb, factor=args.downscale
        )

    # Load block mapping
    block_map = load_block_map(args.label_map)
    print(f"\nLoaded {len(block_map)} block mappings")

    # Create the world
    block_count = create_minecraft_world(
        occupancy,
        semantic_id,
        block_map,
        args.out,
        offset=(args.offset_x, args.offset_y, args.offset_z)
    )

    print("\n" + "=" * 70)
    print("✓ Conversion Complete!")
    print("=" * 70)
    print(f"\nPlaced {block_count} blocks in the world")
    print(f"World location: {os.path.abspath(args.out)}")
    print("\nNext steps:")
    print("1. Open Amulet Editor")
    print("2. Click 'Select World'")
    print(f"3. Navigate to: {os.path.abspath(args.out)}")
    print("4. Enjoy your structure!")


if __name__ == "__main__":
    main()
