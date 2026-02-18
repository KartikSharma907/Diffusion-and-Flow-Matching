# Quick Start Guide for Modal Users (Laptop/No GPU)

This guide is for students who have a laptop without a GPU and want to use Modal for training.

## Prerequisites

- A laptop (Mac, Windows, or Linux)
- No GPU required!
- Modal account (free $500 credits for registered students)

## Setup (5 minutes)

### 1. Set up local environment (CPU-only)

```bash
# Clone the repository
git clone <repo-url>
cd cmu-10799-diffusion

# Run setup script (auto-detects GPU)
./setup-uv.sh                 # Using uv (10-100x faster than pip)

# Activate the environment (name depends on detected hardware)
source .venv-cpu/bin/activate
```

### 2. Set up Modal

```bash
# Install Modal (if not already installed)
pip install modal

# Create account and authenticate
modal token new
```


### 3. Weights & Biases (wandb) logging on Modal

```bash
# Create a Modal secret that holds your API key
modal secret create wandb-api-key WANDB_API_KEY=your_real_key
```

Then attach the secret in `modal_app.py` so Modal passes it into the container:

```python
@app.function(
    ...,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train(...):
    ...
```

### 4. Verify setup

```bash
# Test local environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Should show: CUDA: False (this is expected for CPU-only setup!)

# Test Modal connection
modal app list
# Should show your Modal apps (or empty list if none yet)
```

## Local Workflow (coding & small debugging)

1. Edit code in your favorite editor locally on your laptop
2. Run quick unit tests locally (can use notebooks)
3. Develop and mini debug (e.g. see if the code compiles)

### One-time setup: Download dataset to Modal volume (RECOMMENDED)
```bash
# Download dataset from HuggingFace Hub and cache it in Modal's persistent volume
# This only needs to be done ONCE - the dataset will be reused for all future training runs
modal run modal_app.py --action download
```

This downloads the CelebA dataset and saves it in HuggingFace Arrow format to `/data/celeba` in your Modal volume. After this, all training runs with `from_hub: true` will automatically use the cached version instead of re-downloading, significantly speeding up startup time.

### Training on Modal (cloud GPU)
```bash
# Train DDPM (runs on L40S GPU in the cloud, uses configs/ddpm.yaml)
modal run modal_app.py --action train --method ddpm

# Train with custom settings
modal run modal_app.py --action train --method ddpm --iterations 50000

# Train with a custom config file
modal run modal_app.py --action train --method ddpm --config configs/custom.yaml
```

**Note**: The configs default to `from_hub: true`. If you've run the download step, the cached dataset will be used automatically (no redownload). If you haven't cached it yet, the dataset will be downloaded from HuggingFace Hub on the first run.

**Config options:**
- `from_hub: true` - Uses cached dataset at `/data/celeba` if available, otherwise downloads from HuggingFace Hub
- `from_hub: false` - Loads from traditional folder structure (`/data/celeba/train/images/`)

### Generate samples
```bash
# Generate samples from trained model
modal run modal_app.py --action sample --method ddpm

# With specific checkpoint
modal run modal_app.py --action sample --method ddpm --checkpoint checkpoints/ddpm/ddpm_0050000.pt
```

### Evaluate (using torch-fidelity)
```bash
# Evaluate using torch-fidelity (computes FID and KID)
modal run modal_app.py --action evaluate --method ddpm

# Or use the shell script for convenience (recommended)
./scripts/evaluate_modal_torch_fidelity.sh ddpm checkpoints/ddpm/ddpm_final.pt

# With custom parameters
modal run modal_app.py --action evaluate --method ddpm \
    --num-samples 5000 --metrics fid,kid
```

## Managing Modal Data

Your checkpoints and data are stored in Modal volumes (persistent cloud storage).

```bash
# List all files in your Modal volume
modal volume ls cmu-10799-diffusion-data

# Download a checkpoint to your laptop
modal volume get cmu-10799-diffusion-data checkpoints/ddpm/ddpm_final.pt ./my_checkpoint.pt

# Download generated samples
modal volume get cmu-10799-diffusion-data samples/ddpm_20260103_120000.png ./samples.png
```