"""
Semantic Segmentation Processing Script
Processes all images in a directory using Mask2Former for semantic segmentation.
Based on the Image_Processing_SemNerf.ipynb notebook.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import transformers
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==============================================================================
# Configuration
# ==============================================================================
INPUT_DIR = "./data/images"  # Directory containing input images
OUTPUT_DIR = "./data"  # Directory for output files
MODEL_CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Downscale factors for generating multiple resolutions
DOWNSCALE_FACTORS = [2, 4, 8]


def setup_device():
    """Set up and return the device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch version: {torch.__version__}")
    print(f"Using Transformers version: {transformers.__version__}")
    print(f"Running on device: {device}")
    return device


def load_model_and_processor(model_checkpoint, device):
    """Load the Mask2Former model and processor."""
    print("\nLoading model and processor...")
    processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_checkpoint).to(device)
    print("Model and processor loaded successfully.")
    return processor, model


def generate_panoptic_classes_json(model, output_dir):
    """Generate panoptic_classes.json file with class mappings and colors."""
    print("\nGenerating panoptic_classes.json with Section 7 colors...")
    
    # Get the complete id-to-label mapping
    id2label = model.config.id2label
    max_id = max([int(k) for k in id2label.keys()])
    num_classes = len(id2label)
    
    # Generate colors using the same spectral colormap as in the notebook
    spectral_colors = [plt.cm.nipy_spectral(i / num_classes) for i in range(num_classes)]
    
    # Initialize lists with placeholders
    thing_names = ["unknown"] * (max_id + 1)
    thing_colors = [[0, 0, 0]] * (max_id + 1)
    
    for class_id_str, label in id2label.items():
        idx = int(class_id_str)
        
        # Enforce "person" naming convention
        if label.lower() in ['human', 'person', 'pedestrian']:
            label = "person"
        
        thing_names[idx] = label
        
        # Convert color to RGB integers [0, 255]
        rgba = spectral_colors[idx] if idx < len(spectral_colors) else (0, 0, 0, 1)
        rgb_int = [int(c * 255) for c in rgba[:3]]
        thing_colors[idx] = rgb_int
        
        print(f"  ID {class_id_str} = {label} [{rgb_int}]")
    
    # Construct the final dictionary
    panoptic_data = {
        "thing": thing_names,
        "thing_colors": thing_colors,
        "stuff": [],
        "stuff_colors": []
    }
    
    # Save the JSON
    output_path = os.path.join(output_dir, "panoptic_classes.json")
    with open(output_path, "w") as f:
        json.dump(panoptic_data, f, indent=2)
    
    print(f"Saved panoptic_classes.json to {output_path}")


def process_images(processor, model, input_dir, output_dir, device):
    """Process all images in the input directory and save segmentation maps."""
    # Gather image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()  # Process in order
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # Create output directory
    seg_output_dir = os.path.join(output_dir, "segmentations", "thing")
    os.makedirs(seg_output_dir, exist_ok=True)
    
    max_val = 0
    min_val = 255
    
    for file_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, file_name)
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess and Inference
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process to get semantic map (H, W)
            target_sizes = [image.size[::-1]]  # (Height, Width)
            semantic_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )[0]
            
            # Convert to numpy uint8
            semantic_map_np = semantic_map.cpu().numpy().astype(np.uint8)
            
            # Track min/max values
            max_val = max(max_val, semantic_map_np.max())
            min_val = min(min_val, semantic_map_np.min())
            
            # Save as Grayscale PNG
            output_filename = os.path.splitext(file_name)[0] + ".png"
            output_path = os.path.join(seg_output_dir, output_filename)
            Image.fromarray(semantic_map_np, mode='L').save(output_path)
            
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            continue
    
    print(f"\nAll images processed and saved to {seg_output_dir}")
    print(f"Max class ID: {max_val}, Min class ID: {min_val}")


def create_downscaled_versions(output_dir, factors):
    """Create downscaled versions of the segmentation maps."""
    base_sem_dir = os.path.join(output_dir, "segmentations", "thing")
    
    if not os.path.exists(base_sem_dir):
        print(f"Warning: {base_sem_dir} does not exist. Skipping downscaling.")
        return
    
    # Get all PNG files
    files = sorted([f for f in os.listdir(base_sem_dir) if f.endswith(".png")])
    
    if not files:
        print("Warning: No PNG files found for downscaling.")
        return
    
    for factor in factors:
        out_dir = os.path.join(output_dir, f"segmentations_{factor}", "thing")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nCreating downscaled versions (factor {factor}x) in {out_dir}...")
        
        for fname in tqdm(files, desc=f"Downscaling {factor}x"):
            try:
                img = Image.open(os.path.join(base_sem_dir, fname))
                w, h = img.size
                # Compute new integer size
                new_size = (w // factor, h // factor)
                # Use NEAREST interpolation to preserve class IDs
                img_resized = img.resize(new_size, Image.NEAREST)
                img_resized.save(os.path.join(out_dir, fname))
            except Exception as e:
                print(f"\nError downscaling {fname}: {e}")
                continue


def main():
    """Main execution function."""
    print("=" * 70)
    print("Semantic Segmentation Processing")
    print("=" * 70)
    
    # Validate input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Load model and processor
    processor, model = load_model_and_processor(MODEL_CHECKPOINT, device)
    
    # Generate panoptic_classes.json
    generate_panoptic_classes_json(model, OUTPUT_DIR)
    
    # Process all images
    process_images(processor, model, INPUT_DIR, OUTPUT_DIR, device)
    
    # Create downscaled versions
    create_downscaled_versions(OUTPUT_DIR, DOWNSCALE_FACTORS)
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

