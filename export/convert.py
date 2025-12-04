import os
import numpy as np
from litemapy import Schematic, Region, BlockState

# === Minecraft Block Palette ===
BLOCK_PALETTE = {
    "minecraft:white_wool": (234, 236, 237), "minecraft:orange_wool": (241, 118, 20),
    "minecraft:magenta_wool": (190, 69, 199), "minecraft:light_blue_wool": (58, 175, 217),
    "minecraft:yellow_wool": (249, 199, 35), "minecraft:lime_wool": (112, 185, 25),
    "minecraft:pink_wool": (237, 141, 172), "minecraft:gray_wool": (62, 68, 71),
    "minecraft:light_gray_wool": (142, 142, 134), "minecraft:cyan_wool": (21, 137, 145),
    "minecraft:purple_wool": (121, 42, 172), "minecraft:blue_wool": (53, 57, 157),
    "minecraft:brown_wool": (114, 71, 40), "minecraft:green_wool": (85, 110, 27),
    "minecraft:red_wool": (161, 39, 34), "minecraft:black_wool": (20, 21, 25),
    "minecraft:stone": (125, 125, 125), "minecraft:cobblestone": (130, 130, 130),
    "minecraft:andesite": (136, 136, 136), "minecraft:diorite": (196, 196, 196),
    "minecraft:granite": (154, 110, 98), "minecraft:sand": (219, 211, 160),
    "minecraft:red_sand": (190, 102, 33), "minecraft:gravel": (136, 126, 126),
    "minecraft:oak_planks": (162, 130, 77), "minecraft:spruce_planks": (126, 87, 48),
    "minecraft:birch_planks": (210, 196, 140), "minecraft:jungle_planks": (172, 123, 90),
    "minecraft:acacia_planks": (180, 94, 57), "minecraft:dark_oak_planks": (99, 76, 50),
    "minecraft:mangrove_planks": (102, 30, 25), "minecraft:cherry_planks": (227, 149, 163),
    "minecraft:warped_planks": (43, 104, 99), "minecraft:crimson_planks": (125, 55, 62),
    "minecraft:bricks": (150, 75, 67), "minecraft:clay": (160, 168, 178),
    "minecraft:terracotta": (152, 94, 67), "minecraft:white_terracotta": (209, 178, 161),
    "minecraft:orange_terracotta": (161, 83, 37), "minecraft:magenta_terracotta": (149, 88, 108),
    "minecraft:light_blue_terracotta": (113, 108, 137), "minecraft:yellow_terracotta": (186, 133, 35),
    "minecraft:lime_terracotta": (103, 117, 52), "minecraft:pink_terracotta": (161, 78, 78),
    "minecraft:gray_terracotta": (57, 42, 35), "minecraft:light_gray_terracotta": (135, 106, 97),
    "minecraft:cyan_terracotta": (86, 91, 91), "minecraft:purple_terracotta": (118, 70, 86),
    "minecraft:blue_terracotta": (74, 60, 91), "minecraft:brown_terracotta": (77, 51, 35),
    "minecraft:green_terracotta": (76, 82, 42), "minecraft:red_terracotta": (143, 61, 46),
    "minecraft:black_terracotta": (37, 23, 16), "minecraft:white_concrete": (207, 213, 214),
    "minecraft:orange_concrete": (224, 97, 0), "minecraft:magenta_concrete": (169, 48, 159),
    "minecraft:light_blue_concrete": (36, 137, 199), "minecraft:yellow_concrete": (241, 175, 21),
    "minecraft:lime_concrete": (94, 168, 24), "minecraft:pink_concrete": (214, 101, 143),
    "minecraft:gray_concrete": (54, 57, 61), "minecraft:light_gray_concrete": (125, 125, 115),
    "minecraft:cyan_concrete": (21, 119, 136), "minecraft:purple_concrete": (100, 31, 156),
    "minecraft:blue_concrete": (44, 46, 143), "minecraft:brown_concrete": (95, 58, 31),
    "minecraft:green_concrete": (73, 91, 36), "minecraft:red_concrete": (142, 32, 29),
    "minecraft:black_concrete": (8, 10, 15)
}

BLOCK_NAMES = np.array(list(BLOCK_PALETTE.keys()))
BLOCK_COLORS = np.array(list(BLOCK_PALETTE.values()), dtype=np.float32)


def map_colors_to_block_names(rgb_colors):
    inputs = rgb_colors[:, None, :].astype(np.float32)
    palette = BLOCK_COLORS[None, :, :]
    dist2 = np.sum((inputs - palette) ** 2, axis=2)
    nearest_indices = np.argmin(dist2, axis=1)
    return BLOCK_NAMES[nearest_indices]


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(_here, "voxel_grid_export.npz")
    output_path = os.path.join(_here, "voxel_structure.litematic")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    # Grid is typically (X, Y, Z_up) from scientific libraries
    occupancy_grid = data['occupancy_grid']
    color_grid = data['color_grid']

    # Get raw indices
    raw_x, raw_y, raw_z = np.where(occupancy_grid)

    # === CRITICAL FIX: AXIS SWAP ===
    # Scientific Z-up -> Minecraft Y-up
    # We map: Sci_X -> MC_X, Sci_Y -> MC_Z, Sci_Z -> MC_Y
    mc_x = raw_x
    mc_y = raw_z  # Z becomes height (Y)
    mc_z = raw_y  # Y becomes depth (Z)

    # Determine dimensions of the NEW coordinate system
    # We add +1 because indices are 0-based
    size_x = int(mc_x.max()) + 1
    size_y = int(mc_y.max()) + 1
    size_z = int(mc_z.max()) + 1

    print(f"Minecraft Grid Size: {size_x}x{size_y}x{size_z}")

    # Initialize Schematic
    reg = Region(0, 0, 0, size_x, size_y, size_z)
    schem = reg.as_schematic(name="VoxelExport", author="Python", description="Z-Up Fixed")

    # Colors need to be pulled using the ORIGINAL indices
    voxel_colors = color_grid[raw_x, raw_y, raw_z]
    if voxel_colors.max() <= 1.0:
        voxel_colors = (voxel_colors * 255).astype(np.uint8)

    print(f"Mapping {len(mc_x):,} blocks...")
    block_names = map_colors_to_block_names(voxel_colors)

    # Cache blocks to speed up the loop
    unique_names = np.unique(block_names)
    block_cache = {name: BlockState(name) for name in unique_names}

    # Fill Region
    count = 0
    total = len(mc_x)

    for x, y, z, name in zip(mc_x, mc_y, mc_z, block_names):
        # Coordinates must be native Python ints
        reg[int(x), int(y), int(z)] = block_cache[name]

        count += 1
        if count % 10000 == 0:
            print(f"Progress: {count}/{total}...", end="\r")

    print(f"\nSaving to {output_path}...")
    schem.save(output_path)
    print("Done!")