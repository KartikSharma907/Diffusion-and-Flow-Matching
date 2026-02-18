#!/bin/bash
# CMU 10-799: Diffusion & Flow Matching
# Setup script - creates virtual environment and installs dependencies
#
# NOTE: For faster installation, consider using setup-uv.sh instead!
#       uv is 10-100x faster than pip.
#
# Usage:
#   ./setup.sh          # Auto-detect (CPU if no GPU found)
#   ./setup.sh cpu      # Force CPU-only (for Modal users)
#   ./setup.sh cuda118  # CUDA 11.8
#   ./setup.sh cuda121  # CUDA 12.1
#   ./setup.sh cuda129  # CUDA 12.9
#   ./setup.sh cuda126  # CUDA 12.6
#   ./setup.sh rocm     # AMD ROCm

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  CMU 10-799: Diffusion & Flow Matching"
echo "  Environment Setup"
echo "=============================================="
echo ""

# Determine the environment type
ENV_TYPE=${1:-"auto"}

if [ "$ENV_TYPE" = "auto" ]; then
    # Auto-detect GPU
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [ -n "$CUDA_VERSION" ]; then
            MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
            MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
            if [ "$MAJOR" -ge 12 ]; then
                if [ "$MINOR" -ge 9 ]; then
                    ENV_TYPE="cuda129"
                elif [ "$MINOR" -ge 6 ]; then
                    ENV_TYPE="cuda126"
                else
                    ENV_TYPE="cuda121"
                fi
            else
                ENV_TYPE="cuda118"
            fi
            echo -e "${GREEN}Detected NVIDIA GPU with CUDA $CUDA_VERSION${NC}"
        else
            ENV_TYPE="cpu"
            echo -e "${YELLOW}nvidia-smi found but couldn't detect CUDA version, using CPU${NC}"
        fi
    elif [ -d "/opt/rocm" ]; then
        ENV_TYPE="rocm"
        echo -e "${GREEN}Detected AMD ROCm installation${NC}"
    else
        ENV_TYPE="cpu"
        echo -e "${YELLOW}No GPU detected, using CPU-only setup${NC}"
    fi
fi

echo -e "Environment type: ${GREEN}$ENV_TYPE${NC}"
echo ""

# Set requirements file based on environment type
case $ENV_TYPE in
    cpu)
        REQ_FILE="environments/requirements-cpu.txt"
        ;;
    cuda118)
        REQ_FILE="environments/requirements-cuda118.txt"
        ;;
    cuda121)
        REQ_FILE="environments/requirements-cuda121.txt"
        ;;
    cuda129)
        REQ_FILE="environments/requirements-cuda129.txt"
        ;;
    cuda126)
        REQ_FILE="environments/requirements-cuda126.txt"
        ;;
    rocm)
        REQ_FILE="environments/requirements-rocm.txt"
        ;;
    *)
        echo -e "${RED}Unknown environment type: $ENV_TYPE${NC}"
        echo "Valid options: cpu, cuda118, cuda121, cuda126, cuda129, rocm"
        exit 1
        ;;
esac

# Check if requirements file exists
if [ ! -f "$REQ_FILE" ]; then
    echo -e "${RED}Requirements file not found: $REQ_FILE${NC}"
    exit 1
fi

# Create virtual environment
VENV_DIR=".venv-${ENV_TYPE}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing environment. To reinstall, run:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  pip install -r $REQ_FILE"
        exit 0
    fi
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from $REQ_FILE..."
echo "This may take a few minutes..."
echo ""
pip install -r "$REQ_FILE"

# Verify installation
echo ""
echo "=============================================="
echo "  Verifying installation..."
echo "=============================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo -e "${GREEN}=============================================="
echo "  Setup complete!"
echo "==============================================${NC}"
echo ""
echo "To activate this environment in the future, run:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo ""
if [ "$ENV_TYPE" = "cpu" ]; then
    echo "For GPU training, use Modal:"
    echo ""
    echo "  modal token new              # First time only"
    echo "  modal run modal_app.py --action train --method ddpm"
    echo ""
fi
echo "To deactivate, run:"
echo ""
echo "  deactivate"
echo ""
