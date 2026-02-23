#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script
# =============================================================================
#
# Usage:
#   ./scripts/evaluate_with_fidelity.sh \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --method ddpm \
#       --dataset-path data/celeba \
#       --metrics kid
#
# =============================================================================

set -e

# Capture Python + fidelity binary from the caller's active env.
PYTHON="$(command -v python)"
FIDELITY_BIN="$(dirname "$PYTHON")/fidelity"
if [[ ! -x "$FIDELITY_BIN" ]]; then
  echo "ERROR: 'fidelity' not found at $FIDELITY_BIN"
  echo "  Activate your env and run: pip install torch-fidelity"
  exit 1
fi

# Defaults
METHOD="flow_map_matching" # (right now you only have ddpm but you will be implementing more methods as hw progresses)
CHECKPOINT="/scr/kartiksh/Diffusion_flow_matching/logs/flow_map_matching_psc_run/flow_map_matching_final.pt"
DATASET_PATH="data/celeba-subset/train/images"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1
SAMPLER="heun"     # heun (midpoint-RK2, 2 NFE/step) or euler
SCHEDULE="uniform" # uniform (recommended) or karras (only useful at N>~10 with small rho)
KARRAS_RHO=3.0     # exponent for karras schedule; avoid rho>4 at low N
GENERATED_DIR=""  # Will be set based on checkpoint location
CACHE_DIR=""      # Will be set based on checkpoint location
GPU="3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --schedule) SCHEDULE="$2"; shift 2 ;;
        --karras-rho) KARRAS_RHO="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Set output directories based on checkpoint location
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
GENERATED_DIR="${CHECKPOINT_DIR}/${METHOD}_samples_${NUM_STEPS}steps/generated"
CACHE_DIR="${CHECKPOINT_DIR}/${METHOD}_samples_${NUM_STEPS}steps/cache"

echo "=========================================="
echo "Torch-Fidelity Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Method: $METHOD"
echo "Dataset: $DATASET_PATH"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Sampler: $SAMPLER"
echo "Schedule: $SCHEDULE (rho=$KARRAS_RHO)"
echo "Output: $GENERATED_DIR"
echo "=========================================="

# Step 1: Generate samples
echo ""
echo "[1/2] Generating samples..."
rm -rf "$GENERATED_DIR"

SAMPLE_CMD="CUDA_VISIBLE_DEVICES=$GPU $PYTHON sample.py \
    --checkpoint $CHECKPOINT \
    --method $METHOD \
    --output_dir $GENERATED_DIR \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --sampler $SAMPLER \
    --schedule $SCHEDULE \
    --karras_rho $KARRAS_RHO"

[ -n "$NUM_STEPS" ] && SAMPLE_CMD="$SAMPLE_CMD --num_steps $NUM_STEPS"

eval $SAMPLE_CMD

# Step 2: Run fidelity
echo ""
echo "[2/2] Computing metrics..."
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

FIDELITY_CMD="CUDA_VISIBLE_DEVICES=$GPU python -m torch_fidelity --gpu 0 --batch-size $BATCH_SIZE --cache-root $CACHE_DIR \
    --input1 $GENERATED_DIR --input2 $DATASET_PATH"

[[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --fid"
[[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --kid"
[[ "$METRICS" == *"is"*  ]] && FIDELITY_CMD="$FIDELITY_CMD --isc"

# Log metrics and save
METRICS_OUTPUT="${CACHE_DIR}/metrics.txt"
{
  echo "Metrics for $METHOD with $NUM_STEPS steps:"
  echo "$FIDELITY_CMD"
} > "$METRICS_OUTPUT"

eval "$FIDELITY_CMD" | tee -a "$METRICS_OUTPUT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
