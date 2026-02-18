#!/bin/bash
#SBATCH --job-name=cmu-10799-diffusion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# SLURM Job Script for CMU 10799 Diffusion & Flow Matching Homework
# =============================================================================
# 
# Usage:
#   sbatch scripts/train.sh ddpm
#   sbatch scripts/train.sh flow_matching
#   sbatch scripts/train.sh ddpm configs/ddpm.yaml --resume checkpoints/ddpm_50000.pt
#   sbatch scripts/train.sh ddpm --overfit-single-batch
#
# Customize the #SBATCH directives above for your cluster:
#   --partition: Your GPU partition name (e.g., gpu, dgx, etc.)
#   --gres: GPU specification (e.g., gpu:a100:1, gpu:v100:1)
#   --time: Maximum wall time
#   --mem: Memory allocation
#
# =============================================================================

# Parse arguments
METHOD=${1:-ddpm}
if [ $# -gt 0 ]; then
    shift
fi

CONFIG_FILE=""
if [ $# -gt 0 ]; then
    case "$1" in
        *.yaml|*.yml)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
fi

echo "=============================================="
echo "CMU 10799 Diffusion & Flow Matching Homework Training Job"
echo "=============================================="
echo "Method: $METHOD"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config: $CONFIG_FILE"
fi
if [ $# -gt 0 ]; then
    echo "Extra args: $*"
fi
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Create log directory
mkdir -p logs

# Load modules (customize for your cluster)
# Common module names - uncomment/modify as needed:
# module load cuda/12.1
# module load cudnn/8.9
# module load python/3.11

# Activate virtual environment
# Check for all possible venv directories
if [ -d ".venv-cuda129" ]; then
    echo "Using .venv-cuda129"
    source .venv-cuda129/bin/activate
elif [ -d ".venv-cuda126" ]; then
    echo "Using .venv-cuda126"
    source .venv-cuda126/bin/activate
elif [ -d ".venv-cuda121" ]; then
    echo "Using .venv-cuda121"
    source .venv-cuda121/bin/activate
elif [ -d ".venv-cuda118" ]; then
    echo "Using .venv-cuda118"
    source .venv-cuda118/bin/activate
elif [ -d ".venv-rocm" ]; then
    echo "Using .venv-rocm"
    source .venv-rocm/bin/activate
elif [ -d ".venv-cpu" ]; then
    echo "Using .venv-cpu"
    source .venv-cpu/bin/activate
elif [ -d ".venv" ]; then
    echo "Using .venv"
    source .venv/bin/activate
else
    echo "ERROR: No virtual environment found. Please set up environment first."
    echo "Run: ./setup-uv.sh or ./setup.sh"
    exit 1
fi

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run training
echo "Starting training..."
# Use method-specific config file
if [ -z "$CONFIG_FILE" ]; then
    if [ "$METHOD" = "ddpm" ]; then
        CONFIG_FILE="configs/ddpm.yaml"
    elif [ "$METHOD" = "flow_matching" ]; then
        CONFIG_FILE="configs/flow_matching.yaml"
    else
        CONFIG_FILE="configs/default.yaml"
    fi
fi

# Read device/num_gpus from config to decide whether to use torchrun
read -r CONFIG_DEVICE CONFIG_NUM_GPUS <<<"$(python - "$CONFIG_FILE" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

infra = config.get("infrastructure", {})
device = infra.get("device", "cuda")
num_gpus = infra.get("num_gpus", None)
if num_gpus is None:
    num_gpus = 1 if device != "cpu" else 0

print(device, num_gpus)
PY
)"

# If CUDA_VISIBLE_DEVICES is set, cap num_gpus to what's visible
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra DEVICES <<<"$CUDA_VISIBLE_DEVICES"
    VISIBLE_GPUS=${#DEVICES[@]}
    if [ "$VISIBLE_GPUS" -gt 0 ] && [ "$CONFIG_NUM_GPUS" -gt "$VISIBLE_GPUS" ]; then
        echo "Warning: config requests $CONFIG_NUM_GPUS GPUs but only $VISIBLE_GPUS visible; using $VISIBLE_GPUS."
        CONFIG_NUM_GPUS=$VISIBLE_GPUS
    fi
fi

# Choose launcher
TORCHRUN_BIN="torchrun"
if ! command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN="python -m torch.distributed.run"
fi

if [ "$CONFIG_DEVICE" != "cpu" ] && [ "$CONFIG_NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training with $CONFIG_NUM_GPUS GPUs..."
    $TORCHRUN_BIN \
        --standalone \
        --nproc_per_node="$CONFIG_NUM_GPUS" \
        train.py \
        --method "$METHOD" \
        --config "$CONFIG_FILE" \
        "$@"
else
    python train.py \
        --method "$METHOD" \
        --config "$CONFIG_FILE" \
        "$@"
fi

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "=============================================="
