#!/bin/bash
set -e

echo "=========================================="
echo "Worldcraft Pipeline Setup"
echo "=========================================="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment 'worldcraft'..."
conda env create -f environment.yml

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate worldcraft

# Install CUDA toolkit
# NOTE: CUDA toolkit version must match PyTorch CUDA version in environment.yml
#       PyTorch cu118 = CUDA 11.8, cu117 = CUDA 11.7, etc.
#       To change CUDA version: update both environment.yml (torch version) and this line
echo ""
echo "Installing CUDA toolkit 11.8..."
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Pre-download models
echo ""
echo "=========================================="
echo "Pre-downloading required models..."
echo "=========================================="
python scripts/download_models.py

# Verify nerfstudio installation
echo ""
echo "Verifying nerfstudio installation..."
ns-train --help > /dev/null 2>&1 && echo "✓ nerfstudio installed successfully" || echo "✗ nerfstudio installation failed"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/images
mkdir -p outputs
mkdir -p exports

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate worldcraft"
echo ""
echo "To run the pipeline:"
echo "  python orchestrate.py --input data/my_scene --output outputs --scene-name my_scene"
echo ""
