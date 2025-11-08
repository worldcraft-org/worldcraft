import numpy as np
import json
import os

def create_test_voxel_data(output_dir="test_data/sample_scene"):
    """Create a small test voxel grid"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 16x16x16 voxel grid
    size = 16
    
    # Occupancy: 1 = filled, 0 = empty
    occupancy = np.zeros((size, size, size), dtype=np.uint8)
    
    # Create a simple house structure
    # Floor (y=0)
    occupancy[:, 0, :] = 1
    
    # Walls (hollow box)
    occupancy[0, :5, :] = 1  # Wall 1
    occupancy[-1, :5, :] = 1  # Wall 2
    occupancy[:, :5, 0] = 1  # Wall 3
    occupancy[:, :5, -1] = 1  # Wall 4
    
    # Roof (y=5)
    occupancy[:, 5, :] = 1
    
    # Windows (cut holes in walls)
    occupancy[0, 2, 5:8] = 1
    occupancy[-1, 2, 5:8] = 1
    
    # RGB colors (random for now)
    rgb = np.random.randint(0, 255, (size, size, size, 3), dtype=np.uint8)
    
    # Semantic IDs
    # 0=empty/air, 1=floor, 2=wall, 3=roof, 4=window, 5=road
    semantic_id = np.zeros((size, size, size), dtype=np.uint8)
    semantic_id[:, 0, :] = 1  # floor
    semantic_id[0, 1:5, :] = 2  # walls
    semantic_id[-1, 1:5, :] = 2
    semantic_id[:, 1:5, 0] = 2
    semantic_id[:, 1:5, -1] = 2
    semantic_id[:, 5, :] = 3  # roof
    semantic_id[0, 2, 5:8] = 4  # windows
    semantic_id[-1, 2, 5:8] = 4
    
    # Save arrays
    np.save(os.path.join(output_dir, "occupancy.npy"), occupancy)
    np.save(os.path.join(output_dir, "rgb.npy"), rgb)
    np.save(os.path.join(output_dir, "semantic_id.npy"), semantic_id)
    
    # Create meta.json
    meta = {
        "dimensions": [size, size, size],
        "labels": {
            "0": "air",
            "1": "floor",
            "2": "wall",
            "3": "roof",
            "4": "window",
            "5": "road"
        }
    }
    
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Test data created in {output_dir}")
    print(f"Shape: {occupancy.shape}")
    print(f"Occupied voxels: {np.sum(occupancy)}")

if __name__ == "__main__":
    create_test_voxel_data()