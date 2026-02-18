# Setup Guide

This guide covers how to set up and run `cmu-10799-diffusion` on different platforms.

## Table of Contents
- [Quick Start](#quick-start)
- [Step-by-Step Setup](#step-by-step-setup)
  - [Option A: Using setup scripts (Recommended - Auto-detect)](#option-a-using-setup-scripts-recommended---auto-detect)
  - [Option B: Manual setup with uv](#option-b-manual-setup-with-uv)
  - [Option C: Manual setup with pip + venv](#option-c-manual-setup-with-pip--venv)
- [Modal (Recommended for class)](#modal)
- [AWS](#aws)
- [SLURM Clusters](#slurm-clusters)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**First, figure out your setup:**

| Your Hardware | Requirements File |
|---------------|-------------------|
| Laptop (no GPU) | `requirements-cpu.txt` |
| macOS (any) | `requirements-cpu.txt` |
| NVIDIA GPU + CUDA 11.x | `requirements-cuda118.txt` |
| NVIDIA GPU + CUDA 12.0-12.5 | `requirements-cuda121.txt` |
| NVIDIA GPU + CUDA 12.6-12.8 | `requirements-cuda126.txt` |
| NVIDIA GPU + CUDA 12.9+ | `requirements-cuda129.txt` |
| AMD GPU | `requirements-rocm.txt` |

**Check your CUDA version** (if you have an NVIDIA GPU):
```bash
nvidia-smi  # Look at top right for "CUDA Version: XX.X"
```

---

## Step-by-Step Setup

### Option A: Using setup scripts (Recommended - Auto-detect)

The easiest way to get started is using the setup scripts that auto-detect your hardware:

#### Step 1: Clone the repository

```bash
git clone <repo-url>
cd cmu-10799-diffusion
```

#### Step 2: Run the setup script

```bash
# Run setup script (auto-detects GPU)
./setup-uv.sh                 # Using uv (10-100x faster than pip)
# or
./setup.sh                    # Using standard pip
```

The setup scripts will:
- Auto-detect your GPU (NVIDIA/AMD) or use CPU if none found
- Create a named virtual environment (`.venv-cpu`, `.venv-cuda121`, etc.)
- Install the appropriate PyTorch version
- Verify the installation

#### Step 3: Activate the environment

```bash
# Activate the environment (name depends on detected hardware)
source .venv-cpu/bin/activate        # If CPU was detected
# or
source .venv-cuda121/bin/activate    # If CUDA 12.1 was detected
```

**Manual environment selection:**
```bash
./setup-uv.sh cpu      # Force CPU-only (for Modal users)
./setup-uv.sh cuda121  # Force CUDA 12.1
./setup-uv.sh cuda126  # Force CUDA 12.6
./setup-uv.sh cuda129  # Force CUDA 12.9
```

**Multiple environments:**
You can have different environments on the same machine:
```bash
./setup-uv.sh cpu      # Creates .venv-cpu
./setup-uv.sh cuda121  # Creates .venv-cuda121

# Switch between them:
source .venv-cpu/bin/activate      # For CPU work
source .venv-cuda121/bin/activate  # For GPU work
```

---

### Option B: Manual setup with uv

[uv](https://github.com/astral-sh/uv) is a fast, modern Python package manager (10-100x faster than pip).

#### Step 1: Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or with pip:**
```bash
pip install uv
```

#### Step 2: Clone the repository

```bash
git clone <repo-url>
cd cmu-10799-diffusion
```

#### Step 3: Create named virtual environment

```bash
uv venv .venv-cpu              # For CPU
# or
uv venv .venv-cuda121          # For CUDA 12.1
```

#### Step 4: Activate the environment

**macOS/Linux:**
```bash
source .venv-cpu/bin/activate        # For CPU
# or
source .venv-cuda121/bin/activate    # For GPU
```

**Windows:**
```powershell
.venv-cpu\Scripts\Activate.ps1       # PowerShell
# or
.venv-cpu\Scripts\activate.bat       # Command Prompt
```

#### Step 5: Install dependencies

Choose the command for your hardware:

```bash
# Laptop / macOS / Modal users (no GPU)
uv pip install -r environments/requirements-cpu.txt

# NVIDIA GPU with CUDA 12.1-12.5 (most common)
uv pip install -r environments/requirements-cuda121.txt

# NVIDIA GPU with CUDA 12.6-12.8
uv pip install -r environments/requirements-cuda126.txt

# NVIDIA GPU with CUDA 12.9+
uv pip install -r environments/requirements-cuda129.txt

# AMD GPU with ROCm
uv pip install -r environments/requirements-rocm.txt
```

#### Step 6: Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
- **With GPU:** `PyTorch 2.x.x, CUDA: True`
- **Without GPU:** `PyTorch 2.x.x, CUDA: False`

---

### Option C: Manual setup with pip + venv

Check your CUDA version: `nvidia-smi` (top right shows "CUDA Version: XX.X")

#### Step 1: Clone the repository

```bash
git clone <repo-url>
cd cmu-10799-diffusion
```

#### Step 2: Create named virtual environment

```bash
python -m venv .venv-cpu        # For CPU
# or
python -m venv .venv-cuda121    # For example, for CUDA 12.1
```

(Use `python3` instead of `python` if needed on your system.)

#### Step 3: Activate the environment

**macOS/Linux:**
```bash
source .venv-cpu/bin/activate        # For CPU
# or
source .venv-cuda121/bin/activate    # For GPU
```

**Windows:**
```powershell
.venv-cpu\Scripts\Activate.ps1       # PowerShell
# or
.venv-cpu\Scripts\activate.bat       # Command Prompt
```

#### Step 4: Install dependencies

Choose the command for your hardware:

```bash
# Laptop / macOS / Modal users (no GPU)
pip install -r environments/requirements-cpu.txt

# NVIDIA GPU with CUDA 12.1-12.5 (most common)
pip install -r environments/requirements-cuda121.txt

# NVIDIA GPU with CUDA 12.6-12.8
pip install -r environments/requirements-cuda126.txt

# NVIDIA GPU with CUDA 12.9+
pip install -r environments/requirements-cuda129.txt

# AMD GPU with ROCm
pip install -r environments/requirements-rocm.txt
```

#### Step 5: Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```