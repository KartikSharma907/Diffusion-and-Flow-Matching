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

# Defaults
METHOD="flow_map_matching" # (right now you only have ddpm but you will be implementing more methods as hw progresses)
CHECKPOINT="/scr/kartiksh/Diffusion_flow_matching/logs/flow_map_matching_psc_run/flow_map_matching_final.pt"
DATASET_PATH="data/celeba-subset/train/images"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1
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
echo "Output: $GENERATED_DIR"
echo "=========================================="

# Step 1: Generate samples
echo ""
echo "[1/2] Generating samples..."
rm -rf "$GENERATED_DIR"

SAMPLE_CMD="CUDA_VISIBLE_DEVICES=$GPU python sample.py \
    --checkpoint $CHECKPOINT \
    --method $METHOD \
    --output_dir $GENERATED_DIR \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE"

[ -n "$NUM_STEPS" ] && SAMPLE_CMD="$SAMPLE_CMD --num_steps $NUM_STEPS"

eval $SAMPLE_CMD

# Step 2: Run fidelity
echo ""
echo "[2/2] Computing metrics..."
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

FIDELITY_CMD="CUDA_VISIBLE_DEVICES=$GPU fidelity --gpu 0 --batch-size $BATCH_SIZE --cache-root $CACHE_DIR \
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
