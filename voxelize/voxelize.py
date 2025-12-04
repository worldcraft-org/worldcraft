import os
import shutil
import tempfile
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import plotly.graph_objects as go

app = FastAPI(title="Point Cloud Voxelizer API")

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

        if self.grid_size is None:
            self.grid_size = self._compute_grid_size(bounds)
        if self.origin is None:
            self.origin = bounds[0]

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

        for i, p in enumerate(points):
            idx = self._point_to_voxel_index(p)
            if idx is None: continue

            x, y, z = idx
            density_grid[x, y, z] += densities[i]
            point_counts[x, y, z] += 1
            occupancy_grid[x, y, z] = True

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

    def _compute_grid_size(self, bounds):
        min_b, max_b = bounds
        extent = max_b - min_b
        return tuple(np.ceil(extent / self.voxel_size).astype(int) + 1)

    def _point_to_voxel_index(self, p):
        rel = p - self.origin
        idx = np.floor(rel / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.grid_size): return None
        return tuple(idx)

    def voxel_index_to_world(self, idx):
        return np.array(idx) * self.voxel_size + self.origin + self.voxel_size / 2

def load_ply_from_temp(ply_path):
    points, colors = None, None
    try:
        import open3d.cpu.pybind.io as o3d_io
        pcd = o3d_io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    except Exception:
        import trimesh
        mesh = trimesh.load(ply_path, process=False)
        points = np.asarray(mesh.vertices)
        if hasattr(mesh.visual, "vertex_colors"):
            colors = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255.0
    
    if points is None or len(points) == 0:
        raise ValueError("Could not load points from file")
        
    return points, colors

@app.post("/visualize", response_class=HTMLResponse)
async def visualize_voxels(
    file: UploadFile = File(...), 
    voxel_size: float = Form(0.05)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_ply_path = tmp.name

    try:
        points, colors = load_ply_from_temp(tmp_ply_path)
        
        data = {
            "points": points,
            "densities": np.ones(len(points)),
            "semantic_labels": np.zeros(len(points), dtype=int),
            "colors": colors,
            "features": None,
            "bounds": (points.min(0), points.max(0)),
        }

        voxelizer = Voxelizer(voxel_size=voxel_size)
        voxel_grid = voxelizer.voxelize(data)

        filled = voxel_grid.occupancy_grid
        x, y, z = np.where(filled)
        
        marker_colors = z
        if voxel_grid.color_grid is not None:
             marker_colors = voxel_grid.color_grid[x, y, z]

        fig = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=5, color=marker_colors, opacity=0.8),
                hoverinfo="skip"
            )
        ])

        fig.update_layout(
            title=f"Voxelized Point Cloud (Size: {voxel_size})",
            scene=dict(aspectmode="cube"),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=True)

    finally:
        if os.path.exists(tmp_ply_path):
            os.remove(tmp_ply_path)

@app.post("/convert-to-npz")
async def convert_to_npz(
    file: UploadFile = File(...), 
    voxel_size: float = Form(0.05)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".ply", ".npz")

    try:
        points, colors = load_ply_from_temp(tmp_in_path)
        
        data = {
            "points": points,
            "densities": np.ones(len(points)),
            "semantic_labels": np.zeros(len(points), dtype=int),
            "colors": colors,
            "features": None,
            "bounds": (points.min(0), points.max(0)),
        }

        voxelizer = Voxelizer(voxel_size=voxel_size)
        voxel_grid = voxelizer.voxelize(data)

        xs, ys, zs = np.where(voxel_grid.occupancy_grid)
        points_out = np.array([
            voxelizer.voxel_index_to_world((x, y, z))
            for x, y, z in zip(xs, ys, zs)
        ])
        
        colors_out = np.full((len(points_out), 3), 150, dtype=np.uint8)
        if voxel_grid.color_grid is not None:
            colors_out = (voxel_grid.color_grid[xs, ys, zs] * 255).astype(np.uint8)

        np.savez_compressed(
            tmp_out_path,
            points=points_out,
            color_grid=colors_out,
            occupancy_grid=voxel_grid.occupancy_grid,
        )
        
        return FileResponse(
            path=tmp_out_path, 
            filename="voxel_grid.npz", 
            media_type='application/octet-stream'
        )

    except Exception as e:
        return {"error": str(e)}


def cli_convert_ply_to_npz(ply_path: str, output_path: str, voxel_size: float = 0.05):
    """
    Command-line interface for converting PLY to NPZ format.
    
    Args:
        ply_path: Path to input PLY file
        output_path: Path to output NPZ file
        voxel_size: Size of voxels (default: 0.05)
    """
    print(f"Converting {ply_path} to {output_path}")
    print(f"Voxel size: {voxel_size}")
    
    try:
        # Load PLY file
        points, colors = load_ply_from_temp(ply_path)
        print(f"Loaded {len(points)} points from PLY file")
        
        # Prepare data
        data = {
            "points": points,
            "densities": np.ones(len(points)),
            "semantic_labels": np.zeros(len(points), dtype=int),
            "colors": colors,
            "features": None,
            "bounds": (points.min(0), points.max(0)),
        }
        
        # Voxelize
        print("Voxelizing...")
        voxelizer = Voxelizer(voxel_size=voxel_size)
        voxel_grid = voxelizer.voxelize(data)
        
        # Extract occupied voxels
        xs, ys, zs = np.where(voxel_grid.occupancy_grid)
        points_out = np.array([
            voxelizer.voxel_index_to_world((x, y, z))
            for x, y, z in zip(xs, ys, zs)
        ])
        
        colors_out = np.full((len(points_out), 3), 150, dtype=np.uint8)
        if voxel_grid.color_grid is not None:
            colors_out = (voxel_grid.color_grid[xs, ys, zs] * 255).astype(np.uint8)
        
        print(f"Voxelized to {len(points_out)} voxels")
        
        # Save NPZ
        np.savez_compressed(
            output_path,
            points=points_out,
            color_grid=colors_out,
            occupancy_grid=voxel_grid.occupancy_grid,
        )
        
        print(f"✓ Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    import argparse
    
    # Check if running as CLI or web server
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Web server mode
        import uvicorn
        print("Starting Voxelizer API server...")
        print("Visit http://localhost:8000/docs for API documentation")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1:
        # CLI mode
        parser = argparse.ArgumentParser(
            description="Convert PLY point cloud to voxelized NPZ format"
        )
        parser.add_argument("input", help="Input PLY file path")
        parser.add_argument("output", help="Output NPZ file path")
        parser.add_argument(
            "--voxel-size",
            type=float,
            default=0.05,
            help="Voxel size (default: 0.05)"
        )
        
        args = parser.parse_args()
        success = cli_convert_ply_to_npz(args.input, args.output, args.voxel_size)
        sys.exit(0 if success else 1)
    else:
        # No arguments - show help
        print("Usage:")
        print("  CLI mode:    python voxelize.py <input.ply> <output.npz> [--voxel-size SIZE]")
        print("  Server mode: python voxelize.py serve")
        sys.exit(1)