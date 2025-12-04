#!/usr/bin/env python3
"""
Pre-download all required models for the Worldcraft pipeline.
This ensures models are cached before running the pipeline.
"""

import os
import sys
from pathlib import Path


def download_mask2former():
    """Download Mask2Former semantic segmentation model."""
    print("\n📦 Downloading Mask2Former model (~1.5GB)...")
    print("   Model: facebook/mask2former-swin-large-mapillary-vistas-semantic")
    
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        
        model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        
        # Download processor
        print("   ↓ Downloading processor...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Download model
        print("   ↓ Downloading model weights...")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        
        print("   ✓ Mask2Former model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"   ✗ Error downloading Mask2Former: {e}")
        return False


def verify_cuda():
    """Verify CUDA is available for PyTorch."""
    print("\n🔍 Verifying CUDA availability...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU count: {torch.cuda.device_count()}")
        else:
            print("\n   ⚠️  WARNING: CUDA not available!")
            print("   The pipeline requires GPU acceleration.")
            print("   Please ensure:")
            print("     1. You have an NVIDIA GPU")
            print("     2. CUDA toolkit is properly installed")
            print("     3. PyTorch was installed with CUDA support")
            return False
        
        print("   ✓ CUDA verification passed")
        return True
        
    except ImportError:
        print("   ✗ Error: PyTorch not installed")
        return False
    except Exception as e:
        print(f"   ✗ Error verifying CUDA: {e}")
        return False


def verify_dependencies():
    """Verify all required packages are installed."""
    print("\n🔍 Verifying dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'fastapi': 'FastAPI',
        'plotly': 'Plotly',
        'open3d': 'Open3D',
        'litemapy': 'LitematicaPy',
        'numpy': 'NumPy',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - MISSING")
            missing.append(name)
    
    # Check nerfstudio separately
    try:
        import nerfstudio
        print(f"   ✓ Nerfstudio (version {nerfstudio.__version__})")
    except ImportError:
        print("   ✗ Nerfstudio - MISSING")
        missing.append('Nerfstudio')
    except AttributeError:
        print("   ✓ Nerfstudio (version unknown)")
    
    if missing:
        print(f"\n   ⚠️  Missing packages: {', '.join(missing)}")
        print("   Please run: conda env create -f environment.yml")
        return False
    
    print("   ✓ All dependencies verified")
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("Worldcraft Pipeline - Model Download & Verification")
    print("=" * 60)
    
    success = True
    
    # Verify dependencies
    if not verify_dependencies():
        success = False
    
    # Verify CUDA
    if not verify_cuda():
        success = False
        print("\n⚠️  Continuing without CUDA - pipeline may fail!")
    
    # Download models
    if not download_mask2former():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All models downloaded and verified successfully!")
        print("=" * 60)
        return 0
    else:
        print("⚠️  Setup completed with warnings/errors")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
