import os
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# === Voxel Grid Dataclass ===
@dataclass
class VoxelGrid:
    grid_size: Tuple[int, int, int]
    voxel_size: float
    origin: np.ndarray
    density_grid: np.ndarray
    semantic_grid: np.ndarray
    color_grid: Optional[np.ndarray]
    feature_grid: Optional[np.ndarray]
    occupancy_grid: np.ndarray

    @property
    def num_voxels(self):
        return np.prod(self.grid_size)

    @property
    def occupied_voxels(self):
        return np.sum(self.occupancy_grid > 0)


# === Voxelizer Class (PLY-only) ===
class Voxelizer:
    def __init__(self, voxel_size: float = 0.5,
                 grid_size: Optional[Tuple[int, int, int]] = None,
                 origin: Optional[np.ndarray] = None):
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.origin = origin
        self.voxel_grid = None

    def voxelize(self, data: Dict) -> VoxelGrid:
        points = data["points"]
        densities = data["densities"]
        semantic_labels = data["semantic_labels"]
        colors = data["colors"]
        features = data["features"]
        bounds = data["bounds"]

        # Compute grid size + origin
        if self.grid_size is None:
            self.grid_size = self._compute_grid_size(bounds)
        if self.origin is None:
            self.origin = bounds[0]

        # Initialize grids
        density_grid = np.zeros(self.grid_size, dtype=np.float32)
        semantic_grid = np.zeros(self.grid_size, dtype=np.int32)
        occupancy_grid = np.zeros(self.grid_size, dtype=bool)

        color_grid = None
        if colors is not None:
            color_grid = np.zeros((*self.grid_size, 3), dtype=np.float32)

        feature_grid = None
        if features is not None:
            feature_dim = features.shape[1]
            feature_grid = np.zeros((*self.grid_size, feature_dim), dtype=np.float32)

        point_counts = np.zeros(self.grid_size, dtype=np.int32)

        # === Voxel loop ===
        for i, p in enumerate(points):
            idx = self._point_to_voxel_index(p)
            if idx is None:
                continue

            x, y, z = idx
            density_grid[x, y, z] += densities[i]
            point_counts[x, y, z] += 1
            occupancy_grid[x, y, z] = True

            # semantic label selection = highest density
            if point_counts[x, y, z] == 1:
                semantic_grid[x, y, z] = semantic_labels[i]
            else:
                avg_d = density_grid[x, y, z] / point_counts[x, y, z]
                if densities[i] > avg_d:
                    semantic_grid[x, y, z] = semantic_labels[i]

            if colors is not None:
                color_grid[x, y, z] += colors[i]
            if features is not None:
                feature_grid[x, y, z] += features[i]

        # === Normalize accumulated voxel values ===
        mask = point_counts > 0
        density_grid[mask] /= point_counts[mask]
        if colors is not None:
            color_grid[mask] /= point_counts[mask][..., None]
        if features is not None:
            feature_grid[mask] /= point_counts[mask][..., None]

        self.voxel_grid = VoxelGrid(
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            origin=self.origin,
            density_grid=density_grid,
            semantic_grid=semantic_grid,
            color_grid=color_grid,
            feature_grid=feature_grid,
            occupancy_grid=occupancy_grid
        )
        return self.voxel_grid

    # === Utility Functions ===
    def _compute_grid_size(self, bounds):
        min_b, max_b = bounds
        extent = max_b - min_b
        return tuple(np.ceil(extent / self.voxel_size).astype(int) + 1)

    def _point_to_voxel_index(self, p):
        rel = p - self.origin
        idx = np.floor(rel / self.voxel_size).astype(int)

        if np.any(idx < 0) or np.any(idx >= self.grid_size):
            return None
        return tuple(idx)

    def voxel_index_to_world(self, idx):
        return np.array(idx) * self.voxel_size + self.origin + self.voxel_size / 2


# === MAIN EXECUTION (PLY-only) ===
if __name__ == "__main__":
    import plotly.graph_objects as go

    print("=== Voxelization from PLY Point Cloud ===\n")

    # point_cloud.ply is in the same folder as voxelize.py
    _here = os.path.dirname(os.path.abspath(__file__))
    ply_path = os.path.join(_here, "point_cloud.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing point cloud file at {ply_path}")

    # --- Load PLY ---
    points, colors = None, None
    try:
        import open3d.cpu.pybind.io as o3d_io
        pcd = o3d_io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"[Open3D] Loaded {len(points):,} points")
    except Exception:
        import trimesh
        mesh = trimesh.load(ply_path, process=False)
        points = np.asarray(mesh.vertices)
        if hasattr(mesh.visual, "vertex_colors"):
            colors = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255.0
        print(f"[Trimesh] Loaded {len(points):,} points")

    if len(points) == 0:
        raise ValueError("Empty point cloud")

    # --- Build input dictionary ---
    data = {
        "points": points,
        "densities": np.ones(len(points)),
        "semantic_labels": np.zeros(len(points), dtype=int),
        "colors": colors,
        "features": None,
        "bounds": (points.min(0), points.max(0)),
    }

    # === Voxelization ===
    voxelizer = Voxelizer(voxel_size=0.005)
    voxel_grid = voxelizer.voxelize(data)

    print(f"Grid size: {voxel_grid.grid_size}")
    print(f"Occupied voxels: {voxel_grid.occupied_voxels:,}")
    print(f"Occupancy ratio: {voxel_grid.occupied_voxels / voxel_grid.num_voxels:.2%}")

    # === Visualization ===
    filled = voxel_grid.occupancy_grid
    x, y, z = np.where(filled)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=3, color=z, colorscale="Viridis"),
            hoverinfo="skip"
        )
    ])

    fig.update_layout(
        title="Voxelized Point Cloud",
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

# Compute voxel center coordinates for occupied voxels
xs, ys, zs = np.where(voxel_grid.occupancy_grid)

points_out = np.array([
    voxelizer.voxel_index_to_world((x, y, z))
    for x, y, z in zip(xs, ys, zs)
])

# Compute voxel colors
if voxel_grid.color_grid is not None:
    colors_out = (voxel_grid.color_grid[xs, ys, zs] * 255).astype(np.uint8)
else:
    colors_out = np.full((len(points_out), 3), 150, dtype=np.uint8)



# === Export voxel grid to NumPy NPZ ===
output_npz = os.path.join(_here, "voxel_grid_export.npz")

np.savez_compressed(
    output_npz,
    points=points_out,                  # (N, 3) voxel center coordinates
    colors=colors_out,                  # (N, 3) uint8 RGB
    occupancy_grid=voxel_grid.occupancy_grid,
    density_grid=voxel_grid.density_grid,
    semantic_grid=voxel_grid.semantic_grid,
    color_grid=voxel_grid.color_grid,
    feature_grid=voxel_grid.feature_grid,
    origin=voxel_grid.origin,
    voxel_size=voxel_grid.voxel_size,
    grid_size=voxel_grid.grid_size
)

print("\nNumPy export complete →", output_npz)








#     # === Export voxel grid to output PLY ===
#     output_path = os.path.join(_here, "voxel_grid_export.ply")

#     xs, ys, zs = np.where(filled)
#     points_out = np.array([
#         voxelizer.voxel_index_to_world((x, y, z))
#         for x, y, z in zip(xs, ys, zs)
#     ])

#     if voxel_grid.color_grid is not None:
#         colors_out = (voxel_grid.color_grid[xs, ys, zs] * 255).astype(np.uint8)
#     else:
#         colors_out = np.full((len(points_out), 3), 150, dtype=np.uint8)

#     header = f"""ply
# format ascii 1.0
# element vertex {len(points_out)}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# """

#     with open(output_path, "w") as f:
#         f.write(header)
#         for (px, py, pz), (r, g, b) in zip(points_out, colors_out):
#             f.write(f"{px} {py} {pz} {r} {g} {b}\n")

#     print("\nPLY export complete →", output_path)












# import os, sys, importlib.util
# import numpy as np
# from typing import Dict, Tuple, Optional, List, Union
# from dataclasses import dataclass

# # --- Load semnerf.py from the SAME folder as voxelize.py ---
# _here = os.path.dirname(os.path.abspath(__file__))
# _sem_path = os.path.join(_here, "semnerf.py")

# if not os.path.exists(_sem_path):
#     raise FileNotFoundError(f"Couldn't find semnerf.py next to voxelize.py at: {_sem_path}")

# import importlib.util
# _spec = importlib.util.spec_from_file_location("semnerf", _sem_path)
# _sem = importlib.util.module_from_spec(_spec)
# _spec.loader.exec_module(_sem)

# # Use the classes from semnerf.py
# SemNeRF = _sem.SemNeRF
# SemNerfData = _sem.SemNerfData




# @dataclass
# class VoxelGrid:
#     """Container for voxelized data"""
#     grid_size: Tuple[int, int, int]  # Grid dimensions (X, Y, Z)
#     voxel_size: float  # Size of each voxel
#     origin: np.ndarray  # Origin point of the grid
#     density_grid: np.ndarray  # Density values per voxel
#     semantic_grid: np.ndarray  # Semantic labels per voxel
#     color_grid: Optional[np.ndarray] = None  # RGB colors per voxel
#     feature_grid: Optional[np.ndarray] = None  # Feature vectors per voxel
#     occupancy_grid: Optional[np.ndarray] = None  # Binary occupancy grid

#     @property
#     def num_voxels(self) -> int:
#         return np.prod(self.grid_size)

#     @property
#     def occupied_voxels(self) -> int:
#         if self.occupancy_grid is not None:
#             return np.sum(self.occupancy_grid > 0)
#         return np.sum(self.density_grid > 0)


# class Voxelizer:
#     """Voxelization engine for Semantic NeRF data"""

#     def __init__(self, voxel_size: float = 0.5,
#                  grid_size: Optional[Tuple[int, int, int]] = None,
#                  origin: Optional[np.ndarray] = None):
#         self.voxel_size = voxel_size
#         self.grid_size = grid_size
#         self.origin = origin
#         self.voxel_grid = None

#     def voxelize_from_semnerf(self, semnerf: Union[SemNeRF, Dict]) -> VoxelGrid:
#         """Main voxelization function that processes SemNeRF data"""
#         if isinstance(semnerf, SemNeRF):
#             if semnerf.data is None:
#                 raise ValueError("SemNeRF has no data. Generate or load data first.")
#             data = semnerf.get_data_for_voxelization()
#         else:
#             data = semnerf

#         points = data['points']
#         densities = data['densities']
#         semantic_labels = data['semantic_labels']
#         colors = data.get('colors', None)
#         features = data.get('features', None)
#         bounds = data.get('bounds', (points.min(axis=0), points.max(axis=0)))

#         # Compute grid parameters if not provided
#         if self.grid_size is None:
#             self.grid_size = self._compute_grid_size(bounds)
#         if self.origin is None:
#             self.origin = bounds[0]

#         # Initialize voxel grids
#         density_grid = np.zeros(self.grid_size, dtype=np.float32)
#         semantic_grid = np.zeros(self.grid_size, dtype=np.int32)
#         occupancy_grid = np.zeros(self.grid_size, dtype=bool)

#         color_grid = None
#         if colors is not None:
#             color_grid = np.zeros((*self.grid_size, 3), dtype=np.float32)

#         feature_grid = None
#         if features is not None:
#             feature_dim = features.shape[1]
#             feature_grid = np.zeros((*self.grid_size, feature_dim), dtype=np.float32)

#         # Count points per voxel for averaging
#         point_counts = np.zeros(self.grid_size, dtype=np.int32)

#         # --- Voxelization Loop ---
#         for i, point in enumerate(points):
#             voxel_idx = self._point_to_voxel_index(point)
#             if voxel_idx is None:
#                 continue
#             x, y, z = voxel_idx

#             density_grid[x, y, z] += densities[i]
#             point_counts[x, y, z] += 1
#             occupancy_grid[x, y, z] = True

#             # Semantic label selection
#             if point_counts[x, y, z] == 1:
#                 semantic_grid[x, y, z] = semantic_labels[i]
#             else:
#                 avg_density = density_grid[x, y, z] / point_counts[x, y, z]
#                 if densities[i] > avg_density:
#                     semantic_grid[x, y, z] = semantic_labels[i]

#             if colors is not None:
#                 color_grid[x, y, z] += colors[i]
#             if features is not None:
#                 feature_grid[x, y, z] += features[i]

#         # --- Averaging accumulated values ---
#         mask = point_counts > 0
#         density_grid[mask] /= point_counts[mask]
#         if colors is not None:
#             color_grid[mask] /= point_counts[mask][..., None]
#         if features is not None:
#             feature_grid[mask] /= point_counts[mask][..., None]

#         self.voxel_grid = VoxelGrid(
#             grid_size=self.grid_size,
#             voxel_size=self.voxel_size,
#             origin=self.origin,
#             density_grid=density_grid,
#             semantic_grid=semantic_grid,
#             color_grid=color_grid,
#             feature_grid=feature_grid,
#             occupancy_grid=occupancy_grid
#         )
#         return self.voxel_grid

#     def _compute_grid_size(self, bounds: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int, int]:
#         min_bound, max_bound = bounds
#         extent = max_bound - min_bound
#         return tuple(np.ceil(extent / self.voxel_size).astype(int) + 1)

#     def _point_to_voxel_index(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
#         relative_pos = point - self.origin
#         voxel_idx = np.floor(relative_pos / self.voxel_size).astype(int)
#         if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.grid_size):
#             return None
#         return tuple(voxel_idx)

#     def voxel_index_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
#         return np.array(voxel_idx) * self.voxel_size + self.origin + self.voxel_size / 2

# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     import os
#     import numpy as np
#     import plotly.graph_objects as go

#     print("=== Voxelization from PLY Point Cloud ===\n")

#     # Path to point cloud file (same directory as voxelize.py)
#     ply_path = os.path.join(_here, "point_cloud.ply")
#     if not os.path.exists(ply_path):
#         raise FileNotFoundError(f"Missing point cloud file at: {ply_path}")

#     # --- Step 1: Load the PLY file safely ---
#     points, colors = None, None
#     try:
#         # Prefer lightweight Open3D CPU bindings
#         import open3d.cpu.pybind.io as o3d_io
#         pcd = o3d_io.read_point_cloud(ply_path)
#         points = np.asarray(pcd.points)
#         colors = np.asarray(pcd.colors) if pcd.has_colors() else None

#         if len(points) == 0:
#             raise ValueError("Open3D read 0 points")

#         print(f"[Open3D] Loaded {len(points):,} points")

#     except Exception as e:
#         print(f"[Open3D WARNING] Failed to load PLY ({e}), trying Trimesh...")
#         import trimesh
#         mesh = trimesh.load(ply_path, process=False)

#         points = np.asarray(mesh.vertices)
#         if hasattr(mesh.visual, "vertex_colors"):
#             colors = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255.0

#         print(f"[Trimesh] Loaded {len(points):,} points")

#     if points is None or len(points) == 0:
#         raise ValueError("No valid points found in point_cloud.ply")

#     # --- Step 2: Wrap data in SemNeRF-compatible dictionary ---
#     data_dict = {
#         "points": points,
#         "densities": np.ones(len(points)),  # uniform density
#         "semantic_labels": np.zeros(len(points), dtype=int),
#         "colors": colors if colors is not None else None,
#         "features": None,
#         "bounds": (points.min(axis=0), points.max(axis=0)),
#     }

#     # --- Step 3: Initialize voxelizer ---
#     voxelizer = Voxelizer(voxel_size=0.05)
#     print("\n2. Voxelizing point cloud...")

#     voxel_grid = voxelizer.voxelize_from_semnerf(data_dict)
#     print(f"   Grid size: {voxel_grid.grid_size}")
#     print(f"   Occupied voxels: {voxel_grid.occupied_voxels:,}")
#     print(f"   Occupancy ratio: {voxel_grid.occupied_voxels / voxel_grid.num_voxels:.2%}")

#     # --- Step 4: Visualize voxelized grid ---
#     print("\n3. Rendering voxelized scene...")
#     semantic_labels = voxel_grid.semantic_grid.copy()
#     filled = voxel_grid.occupancy_grid.copy()
#     semantic_labels[~filled] = -1

#     x, y, z = np.indices(semantic_labels.shape)
#     x, y, z = x.flatten(), y.flatten(), z.flatten()
#     labels = semantic_labels.flatten()

#     # --- Define discrete color palette ---
#     tab10_colors = [
#         "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
#     ]
#     background_color = "#d3d3d3"
#     all_colors = [background_color] + tab10_colors
#     color_indices = np.clip(labels + 1, 0, len(all_colors) - 1)
#     colors = [all_colors[i] for i in color_indices]

#     # --- Build interactive 3D plot ---
#     fig = go.Figure(data=[
#         go.Scatter3d(
#             x=x, y=y, z=z,
#             mode="markers",
#             marker=dict(
#                 size=4,
#                 color=colors,
#                 opacity=1.0,
#                 line=dict(width=0)
#             ),
#             hoverinfo="skip"
#         )
#     ])

#     fig.update_layout(
#         title="Voxelized PLY Point Cloud",
#         scene=dict(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             zaxis=dict(visible=False),
#             aspectmode="cube"
#         ),
#         margin=dict(l=0, r=0, b=0, t=40),
#     )

#     fig.show()
#     print("\n=== Voxelization Complete ===")



# # --- Step 4: Visualize voxelized grid ---
# print("\n3. Rendering voxelized scene...")

# # Occupied and empty voxel masks
# filled = voxel_grid.occupancy_grid
# x, y, z = np.indices(filled.shape)

# # Flatten all coordinates
# x, y, z = x.flatten(), y.flatten(), z.flatten()

# # Normalize z for height-based color
# z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

# # Viridis color map for filled voxels
# from matplotlib import cm
# colormap = cm.get_cmap("viridis")
# color_map_vals = np.array([
#     colormap(v)[:3] for v in z_norm
# ]) * 255

# # Initialize gray base color for all voxels
# colors = np.tile(np.array([[80, 80, 80]]), (len(x), 1))
# opacities = np.full(len(x), 1)   # 🔹 low opacity for gray (empty) cubes

# # Replace with colored + opaque voxels where filled
# mask_flat = filled.flatten()
# colors[mask_flat] = color_map_vals[mask_flat]
# opacities[mask_flat] = 1.0          # 🔹 full opacity for occupied voxels

# # Convert to Plotly RGBA color strings (each voxel gets its own opacity)
# colors_rgba = [
#     f"rgba({int(r)}, {int(g)}, {int(b)}, {a:.2f})"
#     for (r, g, b), a in zip(colors, opacities)
# ]

# # --- Build Plotly figure ---
# fig = go.Figure(data=[
#     go.Scatter3d(
#         x=x, y=y, z=z,
#         mode="markers",
#         marker=dict(
#             size=5,                     # cube size
#             symbol="square",
#             color=colors_rgba,          # 🔹 includes per-voxel alpha
#             line=dict(width=0.2, color="black"),
#         ),
#         hoverinfo="none"
#     )
# ])

# fig.update_layout(
#     title="Rubik’s Cube-Style Voxel Grid (Semi-Transparent Empty Space)",
#     scene=dict(
#         xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         aspectmode="cube"
#     ),
#     paper_bgcolor="black",
#     plot_bgcolor="black",
#     margin=dict(l=0, r=0, b=0, t=40),
# )

# fig.show()
# print("\n=== Voxelization Complete ===")



# # === Step 5: Export voxel grid as a .ply file ===
# print("\n4. Exporting voxel grid to PLY...")

# import numpy as np
# import os

# # Save PLY in the same folder as voxelize.py
# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_ply = os.path.join(script_dir, "voxel_grid_export.ply")

# # Extract occupied voxel coordinates
# filled = voxel_grid.occupancy_grid
# xs, ys, zs = np.where(filled)

# # Convert voxel indices to world coordinates (center of voxel)
# points = np.array([
#     voxelizer.voxel_index_to_world((x, y, z))
#     for x, y, z in zip(xs, ys, zs)
# ])

# # Handle voxel colors if available
# if voxel_grid.color_grid is not None:
#     color_vals = voxel_grid.color_grid[xs, ys, zs]
#     colors = (color_vals * 255).astype(np.uint8)
# else:
#     colors = np.full((len(points), 3), 150, dtype=np.uint8)

# # Build PLY header
# ply_header = f"""ply
# format ascii 1.0
# element vertex {len(points)}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# """

# # Write to PLY file
# with open(output_ply, "w") as f:
#     f.write(ply_header)
#     for (px, py, pz), (r, g, b) in zip(points, colors):
#         f.write(f"{px} {py} {pz} {r} {g} {b}\n")

# print("PLY export complete →", os.path.abspath(output_ply))







# --- Step 4: Visualize voxelized grid ---
# print("\n3. Rendering voxelized scene...")

# # Only occupied voxels
# filled = voxel_grid.occupancy_grid
# x, y, z = np.indices(filled.shape)
# x, y, z = x[filled], y[filled], z[filled]

# # Normalize height (z) for color mapping
# z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

# # Use Viridis colormap for smooth color gradient
# from matplotlib import cm
# colormap = cm.get_cmap("viridis")
# colors = [
#     f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"
#     for r, g, b, _ in colormap(z_norm)
# ]

# # --- Build 3D Rubik’s Cube-Style visualization ---
# fig = go.Figure(data=[
#     go.Scatter3d(
#         x=x, y=y, z=z,
#         mode="markers",
#         marker=dict(
#             size=5,                    # cube size
#             symbol="square",
#             color=colors,
#             line=dict(width=0.2, color="black"),  # cube edges
#             opacity=1.0
#         ),
#         hoverinfo="none"
#     )
# ])

# fig.update_layout(
#     title="Voxelized Point Cloud (Rubik’s Cube Style, Occupied Voxels Only)",
#     scene=dict(
#         xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
#         aspectmode="cube"
#     ),
#     paper_bgcolor="black",
#     plot_bgcolor="black",
#     margin=dict(l=0, r=0, b=0, t=40),
# )

# fig.show()
# print("\n=== Voxelization Complete ===")











