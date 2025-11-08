import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_voxel_data(vox_dir):
    """Load voxel data from directory"""
    occupancy = np.load(f"{vox_dir}/occupancy.npy")
    semantic_id = np.load(f"{vox_dir}/semantic_id.npy")
    rgb = np.load(f"{vox_dir}/rgb.npy")
    
    with open(f"{vox_dir}/meta.json", 'r') as f:
        meta = json.load(f)
    
    return occupancy, semantic_id, rgb, meta

def load_block_map(label_map_path):
    """Load semantic label to block mapping"""
    with open(label_map_path, 'r') as f:
        return json.load(f)

def visualize_structure_3d(occupancy, semantic_id, rgb, meta, block_map):
    """Create an interactive 3D visualization of the structure"""
    print("Creating 3D visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get occupied voxel positions
    x, y, z = np.where(occupancy > 0)
    
    # Get colors for each voxel
    colors = []
    for i, j, k in zip(x, y, z):
        # Use RGB from voxel data (normalized to 0-1)
        r, g, b = rgb[i, j, k] / 255.0
        colors.append([r, g, b, 0.8])  # Add alpha
    
    # Plot voxels
    ax.scatter(x, y, z, c=colors, marker='s', s=50, edgecolors='black', linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Minecraft Structure Preview (3D Voxel View)', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add statistics
    labels = meta.get('labels', {})
    unique, counts = np.unique(semantic_id[occupancy > 0], return_counts=True)
    
    stats_text = "Block Distribution:\n"
    for label_id, count in zip(unique, counts):
        label_name = labels.get(str(label_id), f"unknown_{label_id}")
        if str(label_id) in block_map:
            block_name = block_map[str(label_id)]['block']
            stats_text += f"  {label_name} ({block_name}): {count}\n"
    
    # Add text box with stats
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"Total voxels displayed: {len(x)}")
    print("✓ Visualization ready! Rotate with mouse, zoom with scroll wheel.")
    
    plt.tight_layout()
    return fig

def visualize_slices(occupancy, semantic_id, rgb, meta, block_map):
    """Create 2D slice views (top, front, side)"""
    print("Creating 2D slice views...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top view (looking down) - max projection along Y axis
    top_occ = np.max(occupancy, axis=1)  # Shape: (X, Z)
    top_img = np.ones((top_occ.shape[0], top_occ.shape[1], 3))  # White background
    for i in range(top_occ.shape[0]):
        for j in range(top_occ.shape[1]):
            if top_occ[i, j]:
                # Get the topmost voxel's color
                y_indices = np.where(occupancy[i, :, j] > 0)[0]
                if len(y_indices) > 0:
                    y_top = y_indices[-1]  # Highest Y
                    top_img[i, j] = rgb[i, y_top, j] / 255.0
    
    axes[0, 0].imshow(top_img, origin='lower')
    axes[0, 0].set_title('Top View (Looking Down)', fontweight='bold')
    axes[0, 0].set_xlabel('Z')
    axes[0, 0].set_ylabel('X')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Front view - max projection along Z axis
    front_occ = np.max(occupancy, axis=2)  # Shape: (X, Y)
    front_img = np.ones((front_occ.shape[1], front_occ.shape[0], 3))  # White background, transposed
    for i in range(front_occ.shape[0]):
        for j in range(front_occ.shape[1]):
            if front_occ[i, j]:
                # Get the frontmost voxel's color
                z_indices = np.where(occupancy[i, j, :] > 0)[0]
                if len(z_indices) > 0:
                    z_front = z_indices[0]  # Lowest Z (front)
                    front_img[j, i] = rgb[i, j, z_front] / 255.0
    
    axes[0, 1].imshow(front_img, origin='lower')
    axes[0, 1].set_title('Front View', fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Side view - max projection along X axis
    side_occ = np.max(occupancy, axis=0)  # Shape: (Y, Z)
    side_img = np.ones((side_occ.shape[0], side_occ.shape[1], 3))  # White background
    for i in range(side_occ.shape[0]):
        for j in range(side_occ.shape[1]):
            if side_occ[i, j]:
                # Get the side voxel's color
                x_indices = np.where(occupancy[:, i, j] > 0)[0]
                if len(x_indices) > 0:
                    x_side = x_indices[0]  # Lowest X (left side)
                    side_img[i, j] = rgb[x_side, i, j] / 255.0
    
    axes[1, 0].imshow(side_img, origin='lower')
    axes[1, 0].set_title('Side View', fontweight='bold')
    axes[1, 0].set_xlabel('Z')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics panel
    axes[1, 1].axis('off')
    labels = meta.get('labels', {})
    unique, counts = np.unique(semantic_id[occupancy > 0], return_counts=True)
    
    stats_text = "STRUCTURE STATISTICS\n" + "="*40 + "\n\n"
    stats_text += f"Dimensions: {occupancy.shape}\n"
    stats_text += f"Total Blocks: {np.sum(occupancy)}\n\n"
    stats_text += "Block Distribution:\n" + "-"*40 + "\n"
    
    for label_id, count in zip(unique, counts):
        label_name = labels.get(str(label_id), f"unknown_{label_id}")
        if str(label_id) in block_map:
            block_name = block_map[str(label_id)]['block']
            percentage = 100 * count / np.sum(occupancy)
            stats_text += f"\n{label_name}:\n"
            stats_text += f"  → {block_name}\n"
            stats_text += f"  → {count} blocks ({percentage:.1f}%)\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Minecraft Structure - 2D Views', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Minecraft structure without Minecraft')
    parser.add_argument('--vox_dir', required=True, help='Path to voxel data directory')
    parser.add_argument('--label_map', required=True, help='Path to semantic label map JSON')
    parser.add_argument('--view', choices=['3d', '2d', 'both'], default='both',
                       help='View type: 3d, 2d, or both (default: both)')
    parser.add_argument('--save', help='Save visualization to file (e.g., preview.png)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Minecraft Structure Viewer (No Minecraft Required!)")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {args.vox_dir}...")
    occupancy, semantic_id, rgb, meta = load_voxel_data(args.vox_dir)
    block_map = load_block_map(args.label_map)
    
    print(f"✓ Loaded structure: {occupancy.shape}")
    print(f"✓ Total blocks: {np.sum(occupancy)}")
    
    # Create visualizations
    if args.view in ['3d', 'both']:
        fig_3d = visualize_structure_3d(occupancy, semantic_id, rgb, meta, block_map)
        if args.save:
            save_path = args.save.replace('.png', '_3d.png')
            fig_3d.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 3D view to: {save_path}")
    
    if args.view in ['2d', 'both']:
        fig_2d = visualize_slices(occupancy, semantic_id, rgb, meta, block_map)
        if args.save:
            save_path = args.save.replace('.png', '_2d.png')
            fig_2d.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 2D views to: {save_path}")
    
    print("\n" + "="*60)
    print("✓ Visualization Complete!")
    print("="*60)
    print("\nControls:")
    print("  - 3D View: Click and drag to rotate, scroll to zoom")
    print("  - Close window when done")
    
    if not args.save:
        plt.show()

if __name__ == "__main__":
    main()