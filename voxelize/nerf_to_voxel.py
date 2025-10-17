"""
NeRF to Voxel Grid Converter
Converts a NeRF scene representation into a dense semantic voxel grid.
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Try importing optional dependencies, but provide fallbacks if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. Using NumPy for computations.")
    torch = None
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not found. Image loading functionality may be limited.")
    Image = None
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not found. Some visualization features may be limited.")
    cv2 = None
    CV2_AVAILABLE = False


class VoxelGrid:
    """Represents a 3D voxel grid with associated metadata."""
    
    def __init__(self, voxel_size, bbox_min, bbox_max):
        """
        Initialize voxel grid parameters.
        
        Args:
            voxel_size: Size of each voxel in meters
            bbox_min: [x_min, y_min, z_min] in world coordinates
            bbox_max: [x_max, y_max, z_max] in world coordinates
        """
        self.voxel_size = voxel_size
        self.bbox_min = np.array(bbox_min)
        self.bbox_max = np.array(bbox_max)
        
        # Calculate grid dimensions
        self.grid_size = self._compute_grid_size()
        
        print(f"Grid size: {self.grid_size}")
        print(f"Total voxels: {np.prod(self.grid_size):,}")
        print(f"Memory estimate: ~{self._estimate_memory_mb():.1f} MB")
    
    def _compute_grid_size(self):
        """Compute number of voxels needed in each dimension."""
        lengths = self.bbox_max - self.bbox_min
        grid_size = np.ceil(lengths / self.voxel_size).astype(int)
        return grid_size
    
    def _estimate_memory_mb(self):
        """Estimate memory usage for all arrays."""
        n_voxels = np.prod(self.grid_size)
        # occupancy (bool=1) + rgb (3*uint8) + semantic (int32=4) = 8 bytes per voxel
        return n_voxels * 8 / (1024 ** 2)
    
    def world_to_voxel(self, coords):
        """
        Convert world coordinates to voxel indices.
        
        Args:
            coords: [..., 3] array of world coordinates
        
        Returns:
            [..., 3] array of voxel indices
        """
        indices = np.floor((coords - self.bbox_min) / self.voxel_size).astype(int)
        return indices
    
    def voxel_to_world(self, indices):
        """
        Convert voxel indices to world coordinates (voxel centers).
        
        Args:
            indices: [..., 3] array of voxel indices
        
        Returns:
            [..., 3] array of world coordinates
        """
        coords = self.bbox_min + (indices + 0.5) * self.voxel_size
        return coords
    
    def get_transform_matrix(self):
        """
        Get 4x4 world-to-voxel transformation matrix.
        
        Returns:
            4x4 numpy array (homogeneous coordinates)
        """
        scale = 1.0 / self.voxel_size
        tx, ty, tz = -self.bbox_min * scale
        
        matrix = np.array([
            [scale, 0,     0,     tx],
            [0,     scale, 0,     ty],
            [0,     0,     scale, tz],
            [0,     0,     0,     1]
        ])
        return matrix
    
    def create_sampling_grid(self):
        """
        Create a 3D grid of world coordinates for sampling.
        
        Returns:
            Grid of shape (X, Y, Z, 3) with world coordinates at each voxel center
        """
        x_indices = np.arange(self.grid_size[0])
        y_indices = np.arange(self.grid_size[1])
        z_indices = np.arange(self.grid_size[2])
        
        # Create meshgrid
        grid_x, grid_y, grid_z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Stack into (X, Y, Z, 3)
        indices = np.stack([grid_x, grid_y, grid_z], axis=-1)
        
        # Convert to world coordinates
        coords = self.voxel_to_world(indices)
        
        return coords


class DummyNeRF:
    """Dummy NeRF model for testing (replace with real model later)."""
    
    def __init__(self, center=None):
        """
        Initialize dummy NeRF.
        
        Args:
            center: [x, y, z] center point for test geometry
        """
        self.center = np.array(center if center is not None else [0, 0, 10])
    
    def query(self, coords):
        """
        Query NeRF at given coordinates.
        
        Args:
            coords: [..., 3] array of world coordinates
        
        Returns:
            density: [...] array of density values
            rgb: [..., 3] array of RGB colors [0, 1]
            semantic_logits: [..., K] array of semantic logits
        """
        shape = coords.shape[:-1]
        
        # Create a sphere centered at self.center
        distances = np.linalg.norm(coords - self.center, axis=-1)
        radius = 5.0
        
        # Density: gaussian falloff
        density = 3.0 * np.exp(-((distances / radius) ** 2))
        
        # RGB: position-based gradient
        rgb = np.zeros((*shape, 3))
        rgb[..., 0] = np.clip((coords[..., 0] + 10) / 20, 0, 1)  # R: x gradient
        rgb[..., 1] = np.clip((coords[..., 1] + 10) / 20, 0, 1)  # G: y gradient
        rgb[..., 2] = np.clip((coords[..., 2]) / 20, 0, 1)       # B: z gradient
        
        # Semantic: simple rule (0=air, 1=building, 2=vegetation)
        semantic_logits = np.zeros((*shape, 3))
        semantic_logits[..., 0] = 5.0  # default to air
        semantic_logits[..., 1] = np.where(density > 1.0, 10.0, 0.0)  # building if dense
        semantic_logits[..., 2] = np.where((density > 0.5) & (density <= 1.0), 8.0, 0.0)  # vegetation
        
        return density, rgb, semantic_logits


class RealNeRF:
    """Real NeRF model implementation for inference on real datasets."""
    
    def __init__(self, dataset_path, device=None):
        """
        Initialize real NeRF model.
        
        Args:
            dataset_path: Path to NeRF dataset
            device: Device to run inference on (default: cpu)
        """
        self.dataset_path = Path(dataset_path)
        self.device = "cpu"  # Simplified to always use CPU
        
        # Detect dataset type
        self.dataset_type = self._detect_dataset_type()
        print(f"Detected dataset type: {self.dataset_type}")
        
        # Load dataset configuration
        try:
            self.config = self._load_config()
            # Load camera parameters
            self.cameras = self._load_cameras()
            # Load or initialize model
            self.model = self._load_model()
            print(f"Successfully loaded dataset: {dataset_path}")
        except Exception as e:
            print(f"Warning: Error loading dataset: {e}")
            print("Falling back to simplified model")
            self.config = {}
            self.cameras = []
            self.model = {'camera_positions': np.array([]), 'dataset_type': self.dataset_type}
        
        print(f"Initialized RealNeRF with dataset: {dataset_path}")
    
    def _detect_dataset_type(self):
        """Detect the type of NeRF dataset."""
        # Check for synthetic dataset
        if (self.dataset_path / "transforms_train.json").exists():
            return "synthetic"
        
        # Check for LLFF dataset
        if (self.dataset_path / "poses_bounds.npy").exists():
            if "real_360" in str(self.dataset_path):
                return "real_360"
            else:
                return "llff"
        
        # Default to synthetic
        return "synthetic"
    
    def _load_config(self):
        """Load dataset configuration based on dataset type."""
        if self.dataset_type == "synthetic":
            return self._load_synthetic_config()
        elif self.dataset_type in ["llff", "real_360"]:
            return self._load_llff_config()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_synthetic_config(self):
        """Load configuration for synthetic datasets."""
        config_path = self.dataset_path / "transforms_train.json"
        if not config_path.exists():
            # Try alternate filenames
            alt_paths = [
                self.dataset_path / "transforms.json",
                self.dataset_path / "transforms_test.json",
                self.dataset_path / "transforms_val.json"
            ]
            for path in alt_paths:
                if path.exists():
                    config_path = path
                    break
        
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find transforms_*.json in {self.dataset_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _load_llff_config(self):
        """Load configuration for LLFF datasets."""
        poses_bounds_path = self.dataset_path / "poses_bounds.npy"
        if not poses_bounds_path.exists():
            raise FileNotFoundError(f"Could not find poses_bounds.npy in {self.dataset_path}")
        
        # Load poses_bounds.npy
        poses_bounds = np.load(poses_bounds_path)
        
        # Extract poses and bounds
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # (N, 3, 5)
        bounds = poses_bounds[:, -2:].reshape([-1, 2])    # (N, 2)
        
        # Convert LLFF format to our format
        config = {
            'poses': poses,
            'bounds': bounds,
            'is_llff': True
        }
        
        # Find image directory
        image_dirs = [
            self.dataset_path / "images",
            self.dataset_path / "images_8"
        ]
        
        for img_dir in image_dirs:
            if img_dir.exists():
                config['image_dir'] = img_dir
                break
        
        return config
    
    def _load_cameras(self):
        """Load camera parameters based on dataset type."""
        if self.dataset_type == "synthetic":
            return self._load_synthetic_cameras()
        elif self.dataset_type in ["llff", "real_360"]:
            return self._load_llff_cameras()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_synthetic_cameras(self):
        """Load camera parameters for synthetic datasets."""
        cameras = []
        
        # Process frames from config
        for frame in self.config.get('frames', []):
            # Extract camera parameters
            camera = {
                'file_path': frame.get('file_path', ''),
                'transform_matrix': np.array(frame.get('transform_matrix', [])),
            }
            
            # Add focal length
            if 'camera_angle_x' in self.config:
                # Calculate focal length from camera angle
                angle_x = self.config['camera_angle_x']
                # Assuming a default resolution of 800x800 if not specified
                W = self.config.get('W', 800)
                H = self.config.get('H', 800)
                
                focal = 0.5 * W / np.tan(0.5 * angle_x)
                camera['fl_x'] = focal
                camera['fl_y'] = focal
                camera['cx'] = W / 2
                camera['cy'] = H / 2
            
            cameras.append(camera)
        
        return cameras
    
    def _load_llff_cameras(self):
        """Load camera parameters for LLFF datasets."""
        cameras = []
        
        # Get image files
        image_dir = self.config.get('image_dir')
        if not image_dir:
            raise ValueError("Image directory not found in LLFF dataset")
        
        image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        
        # Process poses from config
        poses = self.config['poses']
        bounds = self.config['bounds']
        
        for i, (pose, bound) in enumerate(zip(poses, bounds)):
            # Convert LLFF pose to transform matrix
            # LLFF: [R | t] with R being the camera-to-world rotation
            R = pose[:, :3]
            t = pose[:, 3]
            
            # LLFF uses a different coordinate system, convert to our convention
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            # Create camera entry
            camera = {
                'file_path': str(image_files[i]) if i < len(image_files) else "",
                'transform_matrix': transform_matrix,
                'near': bound[0],
                'far': bound[1]
            }
            
            # Add focal length if available
            if i < len(image_files) and image_files[i].exists() and Image is not None:
                try:
                    # Estimate focal length from image size
                    img = Image.open(image_files[i])
                    W, H = img.size
                    
                    # LLFF typically uses fx = fy = 0.5 * W
                    camera['fl_x'] = 0.5 * W
                    camera['fl_y'] = 0.5 * W
                    camera['cx'] = W / 2
                    camera['cy'] = H / 2
                except Exception as e:
                    print(f"Warning: Could not load image for camera calibration: {e}")
                    # Use default values
                    camera['fl_x'] = 400
                    camera['fl_y'] = 400
                    camera['cx'] = 400
                    camera['cy'] = 400
            
            cameras.append(camera)
        
        return cameras
    
    def _load_model(self):
        """
        Load NeRF model for inference.
        
        This is a simplified implementation. In a real-world scenario,
        you would load weights from a trained model.
        """
        # Extract camera positions
        camera_positions = []
        for cam in self.cameras:
            if 'transform_matrix' in cam and cam['transform_matrix'] is not None:
                try:
                    # Extract camera position from transform matrix
                    if self.dataset_type == "synthetic":
                        # For synthetic, camera position is in the last row of the inverse transform
                        pos = np.linalg.inv(cam['transform_matrix'])[3, :3]
                    else:
                        # For LLFF, camera position is directly in the translation part
                        pos = cam['transform_matrix'][:3, 3]
                    
                    camera_positions.append(pos)
                except Exception as e:
                    print(f"Warning: Could not extract camera position: {e}")
        
        # Create a proxy for the model
        model = {
            'camera_positions': np.array(camera_positions) if camera_positions else np.array([]),
            'dataset_type': self.dataset_type
        }
        
        return model
    
    def query(self, coords, batch_size=10000):
        """
        Query NeRF at given coordinates.
        
        Args:
            coords: [..., 3] array of world coordinates
            batch_size: Batch size for processing
        
        Returns:
            density: [...] array of density values
            rgb: [..., 3] array of RGB colors [0, 1]
            semantic_logits: [..., K] array of semantic logits
        """
        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 3)
        
        # Initialize output arrays
        density = np.zeros(len(coords_flat))
        rgb = np.zeros((len(coords_flat), 3))
        
        # Process in batches to avoid OOM
        for i in range(0, len(coords_flat), batch_size):
            batch_coords = coords_flat[i:i+batch_size]
            
            # Compute density based on distance to camera positions
            if len(self.model.get('camera_positions', [])) > 0:
                distances = np.min(np.linalg.norm(
                    batch_coords[:, None, :] - self.model['camera_positions'][None, :, :],
                    axis=2
                ), axis=1)
                
                # Adjust max_distance based on dataset type
                if self.dataset_type == "synthetic":
                    max_distance = 2.0
                elif self.dataset_type == "real_360":
                    max_distance = 1.0
                else:  # llff
                    max_distance = 0.5
                
                # Simple density model: higher density closer to cameras
                batch_density = np.maximum(0, 1.0 - distances / max_distance)
            else:
                # Fallback if no camera positions available
                # Create a simple sphere in the center of the volume
                center = np.array([0.0, 0.0, 0.0])
                radius = 0.5
                distances = np.linalg.norm(batch_coords - center, axis=1)
                batch_density = np.maximum(0, 1.0 - distances / radius)
            
            density[i:i+batch_size] = batch_density
            
            # Simple RGB estimation based on position
            batch_rgb = np.zeros((len(batch_coords), 3))
            batch_rgb[:, 0] = np.clip(0.5 + batch_coords[:, 0], 0, 1)  # R
            batch_rgb[:, 1] = np.clip(0.5 + batch_coords[:, 1], 0, 1)  # G
            batch_rgb[:, 2] = np.clip(0.5 + batch_coords[:, 2], 0, 1)  # B
            
            # Apply density as a factor
            batch_rgb *= batch_density[:, None]
            rgb[i:i+batch_size] = batch_rgb
        
        # Reshape outputs to match input shape
        density = density.reshape(original_shape)
        rgb = rgb.reshape((*original_shape, 3))
        
        # For semantic logits, we'll use a simple heuristic
        # In a real implementation, this would come from a semantic segmentation model
        semantic_logits = np.zeros((*original_shape, 3))
        semantic_logits[..., 0] = 5.0  # default to air
        semantic_logits[..., 1] = np.where(density > 0.7, 10.0, 0.0)  # building if dense
        semantic_logits[..., 2] = np.where((density > 0.3) & (density <= 0.7), 8.0, 0.0)  # vegetation
        
        return density, rgb, semantic_logits


class ReplicaNeRF:
    """Loads a Replica dataset and provides density/color query interface."""
    
    def __init__(self, dataset_path=None):
        """
        Initialize with path to a Replica dataset.
        
        Args:
            dataset_path: Path to Replica dataset
        """
        self.dataset_path = dataset_path
        print(f"Loading Replica dataset from: {dataset_path}")
        
        # Check if the download.sh script exists but hasn't been run yet
        if dataset_path is not None:
            download_script = os.path.join(os.path.dirname(str(dataset_path)), "download.sh")
            if os.path.exists(download_script):
                print(f"Note: Replica dataset files need to be downloaded using: {download_script}")
                print("Using simulated Replica room model for now")
        else:
            print("No dataset path provided. Using simulated Replica room model.")
        
        # Load dataset configuration and mesh data
        self._load_config()
        
        # Set up scene bounds
        self._setup_scene_bounds()
    
    def _load_config(self):
        """Load Replica dataset configuration."""
        try:
            # For Replica, we'll create a default configuration
            # In a real implementation, we would parse the PLY files
            self.config = {
                'scene_name': os.path.basename(str(self.dataset_path)) if self.dataset_path else 'replica_room',
                'scale': 1.0
            }
            
            # Check if we have any PLY files
            ply_files = []
            if self.dataset_path and os.path.exists(str(self.dataset_path)):
                for root, dirs, files in os.walk(str(self.dataset_path)):
                    for file in files:
                        if file.endswith('.ply'):
                            ply_files.append(os.path.join(root, file))
            
            if ply_files:
                print(f"Found {len(ply_files)} PLY files in the dataset")
                self.config['ply_files'] = ply_files
                print("Using simulated Replica room model (mesh loading not implemented)")
            else:
                print("Warning: No PLY files found in the dataset")
                print("Using simulated Replica room model")
                self.config['ply_files'] = []
            
        except Exception as e:
            print(f"Error loading Replica config: {e}")
            self.config = {
                'scene_name': 'unknown',
                'scale': 1.0,
                'ply_files': []
            }
    
    def _setup_scene_bounds(self):
        """Set up scene bounds based on available data."""
        try:
            # In a real implementation, we would parse the PLY files to get the bounds
            # For now, we'll use default values
            self.center = np.array([0, 0, 0])
            self.radius = 2.0  # Default radius for Replica scenes
            
            print(f"Scene center: {self.center}")
            print(f"Scene radius: {self.radius}")
            print("Created simulated Replica room model with walls, floor, and furniture")
            
        except Exception as e:
            print(f"Error setting up scene bounds: {e}")
            self.center = np.array([0, 0, 0])
            self.radius = 2.0
    
    def query(self, positions):
        """
        Query the Replica model for density and color at specified positions.
        
        Args:
            positions: [..., 3] array of positions in world space
        
        Returns:
            densities: [...] array of density values
            colors: [..., 3] array of RGB colors
            semantic_logits: [..., N] array of semantic class logits
        """
        # Calculate distance from center
        distances = np.linalg.norm(positions - self.center, axis=-1)
        
        # For Replica, create a room-like structure
        # Walls at the boundary of the radius
        densities = np.zeros_like(distances)
        
        # Create walls at the boundary (0.9 to 1.0 of radius)
        wall_mask = (distances > 0.9 * self.radius) & (distances < self.radius)
        densities[wall_mask] = 10.0  # High density for walls
        
        # Create floor at bottom (negative y)
        floor_mask = (positions[..., 1] < -0.9 * self.radius) & (positions[..., 1] > -self.radius)
        densities[floor_mask] = 10.0  # High density for floor
        
        # Create ceiling at top (positive y)
        ceiling_mask = (positions[..., 1] > 0.9 * self.radius) & (positions[..., 1] < self.radius)
        densities[ceiling_mask] = 10.0  # High density for ceiling
        
        # Create some furniture-like structures in the middle
        furniture_mask = (np.abs(positions[..., 0]) < 0.3 * self.radius) & \
                         (positions[..., 1] < -0.5 * self.radius) & \
                         (positions[..., 1] > -0.9 * self.radius) & \
                         (np.abs(positions[..., 2]) < 0.3 * self.radius)
        densities[furniture_mask] = 5.0  # Medium density for furniture
        
        # Create a table in the center
        table_mask = (np.abs(positions[..., 0]) < 0.4 * self.radius) & \
                     (positions[..., 1] < -0.3 * self.radius) & \
                     (positions[..., 1] > -0.35 * self.radius) & \
                     (np.abs(positions[..., 2]) < 0.4 * self.radius)
        densities[table_mask] = 8.0  # High density for table
        
        # Position-based coloring with structure-specific colors
        colors = np.zeros(positions.shape)
        
        # Wall color (light beige)
        colors[wall_mask] = np.array([0.9, 0.85, 0.7])
        
        # Floor color (wood-like)
        colors[floor_mask] = np.array([0.6, 0.4, 0.2])
        
        # Ceiling color (white)
        colors[ceiling_mask] = np.array([0.95, 0.95, 0.95])
        
        # Furniture color (dark brown)
        colors[furniture_mask] = np.array([0.3, 0.2, 0.1])
        
        # Table color (medium brown)
        colors[table_mask] = np.array([0.5, 0.3, 0.2])
        
        # Clip to [0, 1]
        colors = np.clip(colors, 0, 1)
        
        # Create semantic logits (3 classes: 0=air, 1=structure, 2=furniture)
        semantic_logits = np.zeros((*distances.shape, 3))
        
        # Class 0: Air/void (default)
        semantic_logits[..., 0] = 10.0  # High logit for air by default
        
        # Class 1: Structure (walls, floor, ceiling)
        structure_mask = wall_mask | floor_mask | ceiling_mask
        semantic_logits[structure_mask, 0] = 0.0  # Low logit for air
        semantic_logits[structure_mask, 1] = 10.0  # High logit for structure
        
        # Class 2: Furniture (furniture, table)
        furniture_all_mask = furniture_mask | table_mask
        semantic_logits[furniture_all_mask, 0] = 0.0  # Low logit for air
        semantic_logits[furniture_all_mask, 2] = 10.0  # High logit for furniture
        
        return densities, colors, semantic_logits


class Voxelizer:
    """Main voxelization engine."""
    
    def __init__(self, grid, nerf_model, density_threshold=0.5):
        """
        Initialize voxelizer.
        
        Args:
            grid: VoxelGrid instance
            nerf_model: NeRF model with query() method
            density_threshold: Threshold for occupancy determination
        """
        self.grid = grid
        self.nerf_model = nerf_model
        self.density_threshold = density_threshold
    
    def voxelize(self, chunk_size=None):
        """
        Perform voxelization.
        
        Args:
            chunk_size: If specified, process in chunks to avoid OOM
        
        Returns:
            occupancy: (X, Y, Z) boolean array
            rgb: (X, Y, Z, 3) uint8 array
            semantic_id: (X, Y, Z) int array
        """
        if chunk_size is None:
            # Process entire grid at once
            return self._voxelize_full()
        else:
            # Process in chunks
            return self._voxelize_chunked(chunk_size)
    
    def _voxelize_full(self):
        """Voxelize entire grid at once."""
        print("Creating sampling grid...")
        coords = self.grid.create_sampling_grid()
        
        print("Querying NeRF...")
        density, rgb, semantic_logits = self.nerf_model.query(coords)
        
        print("Processing results...")
        # Determine occupancy
        occupancy = density > self.density_threshold
        
        # Convert RGB to uint8
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        # Determine semantic class (argmax)
        semantic_id = np.argmax(semantic_logits, axis=-1).astype(np.int32)
        
        return occupancy, rgb_uint8, semantic_id
    
    def _voxelize_chunked(self, chunk_size):
        """
        Voxelize in chunks to reduce memory usage.
        
        Args:
            chunk_size: Number of voxels to process at once
            
        Returns:
            occupancy: (X, Y, Z) boolean array
            rgb: (X, Y, Z, 3) uint8 array
            semantic_id: (X, Y, Z) int array
        """
        print("Creating sampling grid...")
        coords = self.grid.create_sampling_grid()
        
        # Get grid dimensions
        grid_shape = coords.shape[:-1]  # (X, Y, Z)
        
        # Initialize output arrays
        occupancy = np.zeros(grid_shape, dtype=bool)
        rgb_uint8 = np.zeros((*grid_shape, 3), dtype=np.uint8)
        semantic_id = np.zeros(grid_shape, dtype=np.int32)
        
        # Flatten coordinates for chunked processing
        coords_flat = coords.reshape(-1, 3)
        total_voxels = len(coords_flat)
        
        print(f"Processing {total_voxels:,} voxels in chunks of {chunk_size:,}...")
        
        # Process in chunks
        for i in range(0, total_voxels, chunk_size):
            end_idx = min(i + chunk_size, total_voxels)
            chunk_coords = coords_flat[i:end_idx]
            
            print(f"  Processing chunk {i//chunk_size + 1}/{(total_voxels + chunk_size - 1)//chunk_size}: "
                  f"voxels {i:,} to {end_idx-1:,}")
            
            # Query NeRF for this chunk
            density, rgb, semantic_logits = self.nerf_model.query(chunk_coords)
            
            # Process results for this chunk
            chunk_occupancy = density > self.density_threshold
            chunk_rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            chunk_semantic_id = np.argmax(semantic_logits, axis=-1).astype(np.int32)
            
            # Map flat indices back to 3D grid
            flat_indices = np.arange(i, end_idx)
            x_indices = flat_indices // (grid_shape[1] * grid_shape[2])
            y_indices = (flat_indices % (grid_shape[1] * grid_shape[2])) // grid_shape[2]
            z_indices = flat_indices % grid_shape[2]
            
            # Update output arrays
            occupancy[x_indices, y_indices, z_indices] = chunk_occupancy
            rgb_uint8[x_indices, y_indices, z_indices] = chunk_rgb_uint8
            semantic_id[x_indices, y_indices, z_indices] = chunk_semantic_id
        
        print("Chunked processing complete.")
        return occupancy, rgb_uint8, semantic_id


def save_outputs(output_dir, grid, occupancy, rgb, semantic_id, label_map):
    """
    Save voxelization outputs to disk.
    
    Args:
        output_dir: Output directory path
        grid: VoxelGrid instance
        occupancy: Occupancy array
        rgb: RGB array
        semantic_id: Semantic ID array
        label_map: Dictionary mapping IDs to labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    
    # Save numpy arrays
    np.save(output_path / "occupancy.npy", occupancy)
    np.save(output_path / "rgb.npy", rgb)
    np.save(output_path / "semantic_id.npy", semantic_id)
    print("✓ Saved .npy files")
    
    # Create metadata
    meta = {
        "scene_id": output_path.name,
        "voxel_size_m": float(grid.voxel_size),
        "bbox_world": {
            "min": grid.bbox_min.tolist(),
            "max": grid.bbox_max.tolist()
        },
        "grid_size": grid.grid_size.tolist(),
        "world_to_voxel_transform": grid.get_transform_matrix().tolist(),
        "coordinate_system": {
            "origin": "bbox_min",
            "axes": "ENU (East-North-Up)",
            "handedness": "right",
            "units": "meters"
        },
        "label_set": label_map,
        "color_encoding": "uint8_rgb",
        "density_threshold": 0.5,
        "creation_date": datetime.now().isoformat(),
        "version": "v0.1_week1_prototype",
        "notes": "Week 1 prototype using dummy NeRF data"
    }
    
    with open(output_path / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("✓ Saved meta.json")
    
    # Statistics
    occupied_count = occupancy.sum()
    total_count = occupancy.size
    print(f"\nStatistics:")
    print(f"  Occupied voxels: {occupied_count:,} / {total_count:,}")
    print(f"  Occupancy rate: {100 * occupied_count / total_count:.2f}%")


def visualize_slices(output_dir, grid, occupancy, rgb, semantic_id, label_map):
    """
    Generate slice visualizations for quick inspection.
    
    Args:
        output_dir: Output directory path
        grid: VoxelGrid instance
        occupancy: Occupancy array
        rgb: RGB array
        semantic_id: Semantic ID array
        label_map: Dictionary mapping IDs to labels
    """
    output_path = Path(output_dir)
    
    print("\nGenerating visualizations...")
    
    # Select middle slices
    mid_x = grid.grid_size[0] // 2
    mid_y = grid.grid_size[1] // 2
    mid_z = grid.grid_size[2] // 2
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # XY plane (top view)
    axes[0, 0].imshow(occupancy[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Occupancy XY (z={mid_z})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(rgb[:, :, mid_z, :].transpose(1, 0, 2), origin='lower')
    axes[0, 1].set_title(f'RGB XY (z={mid_z})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    axes[0, 2].imshow(semantic_id[:, :, mid_z].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[0, 2].set_title(f'Semantic XY (z={mid_z})')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    
    # XZ plane (side view)
    axes[1, 0].imshow(occupancy[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'Occupancy XZ (y={mid_y})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Z')
    
    axes[1, 1].imshow(rgb[:, mid_y, :, :].transpose(1, 0, 2), origin='lower')
    axes[1, 1].set_title(f'RGB XZ (y={mid_y})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    
    axes[1, 2].imshow(semantic_id[:, mid_y, :].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[1, 2].set_title(f'Semantic XZ (y={mid_y})')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Z')
    
    # YZ plane (front view)
    axes[2, 0].imshow(occupancy[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[2, 0].set_title(f'Occupancy YZ (x={mid_x})')
    axes[2, 0].set_xlabel('Y')
    axes[2, 0].set_ylabel('Z')
    
    axes[2, 1].imshow(rgb[mid_x, :, :, :].transpose(1, 0, 2), origin='lower')
    axes[2, 1].set_title(f'RGB YZ (x={mid_x})')
    axes[2, 1].set_xlabel('Y')
    axes[2, 1].set_ylabel('Z')
    
    axes[2, 2].imshow(semantic_id[mid_x, :, :].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[2, 2].set_title(f'Semantic YZ (x={mid_x})')
    axes[2, 2].set_xlabel('Y')
    axes[2, 2].set_ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path / "slices_visualization.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved slices_visualization.png")
    plt.close()


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert NeRF to voxel grid')
    parser.add_argument('--voxel_size', type=float, default=0.15,
                        help='Voxel size in meters (default: 0.15)')
    parser.add_argument('--bbox', type=float, nargs=6, 
                        default=[-10, 10, -10, 10, 0, 20],
                        help='Bounding box: xmin xmax ymin ymax zmin zmax')
    parser.add_argument('--out', type=str, default='outputs/scene_001',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, required=False,
                        help='Path to NeRF dataset directory')
    parser.add_argument('--dataset_type', type=str, choices=['synthetic', 'llff', 'real_360', 'replica'],
                        help='Type of dataset (will be auto-detected if not specified)')
    parser.add_argument('--scene', type=str, help='Scene name within the dataset type')
    parser.add_argument('--use_dummy', action='store_true',
                        help='Use dummy NeRF model instead of real dataset')
    parser.add_argument('--density_threshold', type=float, default=0.5,
                        help='Density threshold for occupancy determination')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing (to avoid OOM)')
    parser.add_argument('--archive_dir', type=str, 
                        default="/Users/citrusfurina/Library/CloudStorage/OneDrive-Personal/University/Clubs/Open project/worldcraft/voxelize/archive",
                        help='Directory containing NeRF datasets')
    parser.add_argument('--replica_dir', type=str,
                        default="./Replica-Dataset",
                        help='Directory containing Replica datasets')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NeRF to Voxel Grid Converter")
    print("="*70)
    
    # Setup label map
    label_map = {
        "0": "air/void",
        "1": "building",
        "2": "vegetation"
    }
    
    # Determine dataset path
    dataset_path = args.dataset
    if dataset_path is None and not args.use_dummy and args.scene:
        # Construct path from archive directory
        archive_dir = Path(args.archive_dir)
        
        if not archive_dir.exists():
            print(f"Warning: Archive directory {archive_dir} does not exist.")
            print("Using dummy NeRF model instead.")
            args.use_dummy = True
            dataset_path = None
        elif args.dataset_type:
            # Use specified dataset type
            dataset_type = args.dataset_type
            if dataset_type == "replica":
                # Handle Replica dataset - always use it even if path doesn't exist
                replica_dir = Path(args.replica_dir)
                dataset_path = replica_dir / args.scene
                print(f"Using Replica dataset path: {dataset_path}")
                # We'll use ReplicaNeRF even if the path doesn't exist
                # It will create a simulated room
            elif dataset_type == "synthetic":
                # Check for nested directory structure
                nested_dir = archive_dir / "nerf_synthetic" / "nerf_synthetic"
                if nested_dir.exists():
                    dataset_path = nested_dir / args.scene
                else:
                    dataset_path = archive_dir / "nerf_synthetic" / args.scene
            elif dataset_type == "llff":
                nested_dir = archive_dir / "nerf_llff_data" / "nerf_llff_data"
                if nested_dir.exists():
                    dataset_path = nested_dir / args.scene
                else:
                    dataset_path = archive_dir / "nerf_llff_data" / args.scene
            elif dataset_type == "real_360":
                nested_dir = archive_dir / "nerf_real_360" / "nerf_real_360"
                if nested_dir.exists():
                    dataset_path = nested_dir / args.scene
                else:
                    dataset_path = archive_dir / "nerf_real_360" / args.scene
                
            if not dataset_path or (not dataset_path.exists() and dataset_type != "replica"):
                print(f"Warning: Dataset path {dataset_path} does not exist.")
                print("Using dummy NeRF model instead.")
                args.use_dummy = True
                dataset_path = None
        else:
            # Try to find the scene in any dataset type
            dataset_types = {"synthetic": "nerf_synthetic", "llff": "nerf_llff_data", "real_360": "nerf_real_360"}
            for dt_key, dt_folder in dataset_types.items():
                # Check for nested directory structure
                nested_dir = archive_dir / dt_folder / dt_folder
                if nested_dir.exists():
                    potential_path = nested_dir / args.scene
                else:
                    potential_path = archive_dir / dt_folder / args.scene
                
                if potential_path.exists():
                    dataset_path = potential_path
                    print(f"Found scene {args.scene} in dataset type {dt_key}")
                    break
            else:
                # Check if it's in Replica dataset
                replica_dir = Path(args.replica_dir)
                potential_path = replica_dir / args.scene
                # For Replica, we'll use it even if the path doesn't exist
                dataset_path = potential_path
                args.dataset_type = "replica"
                print(f"Using Replica dataset for scene {args.scene}")
    elif dataset_path is not None and not Path(dataset_path).exists() and args.dataset_type != "replica":
        print(f"Warning: Dataset path {dataset_path} does not exist.")
        print("Using dummy NeRF model instead.")
        args.use_dummy = True
        dataset_path = None
    
    # Initialize grid
    bbox_min = args.bbox[::2]  # [xmin, ymin, zmin]
    bbox_max = args.bbox[1::2]  # [xmax, ymax, zmax]
    grid = VoxelGrid(args.voxel_size, bbox_min, bbox_max)
    
    # Initialize NeRF model
    if args.dataset_type == 'replica':
        print("Using Replica NeRF model")
        nerf_model = ReplicaNeRF(dataset_path=dataset_path)
    elif args.use_dummy or not dataset_path:
        print("Using dummy NeRF model")
        nerf_model = DummyNeRF(center=[0, 0, 10])
    else:
        try:
            print(f"Using dataset from: {dataset_path}")
            print(f"Dataset path exists: {os.path.exists(dataset_path)}")
            print(f"Dataset path contents: {os.listdir(dataset_path) if os.path.exists(dataset_path) else 'Not found'}")
            
            nerf_model = RealNeRF(dataset_path=dataset_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to dummy NeRF model")
            nerf_model = DummyNeRF(center=[0, 0, 10])
    
    # Initialize voxelizer
    voxelizer = Voxelizer(grid, nerf_model, density_threshold=args.density_threshold)
    
    # Run voxelization
    try:
        if args.batch_size > 0 and not args.use_dummy:
            print(f"Using chunked processing with batch size: {args.batch_size}")
            try:
                occupancy, rgb, semantic_id = voxelizer._voxelize_chunked(args.batch_size)
            except NotImplementedError:
                print("Chunked processing not implemented, falling back to full processing")
                occupancy, rgb, semantic_id = voxelizer.voxelize()
        else:
            occupancy, rgb, semantic_id = voxelizer.voxelize()
        
        # Determine output directory
        output_dir = args.out
        if args.scene and not args.use_dummy:
            output_dir = os.path.join(os.path.dirname(args.out), args.scene)
            os.makedirs(output_dir, exist_ok=True)
        
        # Save outputs
        save_outputs(output_dir, grid, occupancy, rgb, semantic_id, label_map)
        
        # Generate visualizations
        try:
            visualize_slices(output_dir, grid, occupancy, rgb, semantic_id, label_map)
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
        
        print("\n" + "="*70)
        print("✓ Complete! Check outputs in:", output_dir)
        print("="*70)
    except Exception as e:
        print(f"Error during voxelization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()