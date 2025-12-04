import os
import numpy as np
import plotly.graph_objects as go


def visualize_voxel_grid(npz_path):
    """
    Loads a compressed voxel grid NPZ and visualizes it using integer grid coordinates
    (voxel indices) with semantic labels on hover.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        print("Please run the voxelization script first to generate the data.")
        return

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    # Extract grids
    occupancy_grid = data['occupancy_grid']
    semantic_grid = data['semantic_grid']

    # Get integer indices (Grid X, Y, Z) for all occupied voxels
    # These represent the 'width', 'depth', 'height' indices in the grid
    xs, ys, zs = np.where(occupancy_grid)

    # 'colors' are (N, 3) uint8 RGB values
    colors = data['colors']

    # Extract 1D array of labels matching the indices
    # We use the same masking logic so the order aligns with xs, ys, zs
    semantic_labels = semantic_grid[occupancy_grid]

    print(f"Visualizing {len(xs):,} voxels in grid space...")

    # Prepare colors for Plotly (requires 'rgb(r,g,b)' strings or normalized arrays)
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors]

    # Create the 3D Scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=4,
                color=color_strings,
                opacity=1.0,
                symbol='square'
            ),
            # Customizing Hover Data to show Integers
            text=[f"Label: {l}" for l in semantic_labels],
            hovertemplate=(
                    "<b>Voxel Grid Index</b><br>" +
                    "X: %{x:d}<br>" +
                    "Y: %{y:d}<br>" +
                    "Z: %{z:d}<br>" +
                    "<b>%{text}</b>" +
                    "<extra></extra>"  # Hides the secondary box
            )
        )
    ])

    # Beautify layout
    fig.update_layout(
        title=f"Voxel Grid Visualization ({os.path.basename(npz_path)})",
        scene=dict(
            xaxis_title='Grid X',
            yaxis_title='Grid Y',
            zaxis_title='Grid Z',
            aspectmode='data',  # Ensures the 3D scale is 1:1:1
            bgcolor="rgb(240, 240, 240)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


if __name__ == "__main__":
    # Assumes the file is in the same directory as this script
    _here = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_here, "voxel_grid_export.npz")

    visualize_voxel_grid(file_path)