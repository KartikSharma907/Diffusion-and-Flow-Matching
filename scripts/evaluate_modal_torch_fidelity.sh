#!/bin/bash
# =============================================================================
# Modal Torch-Fidelity Evaluation Script
# =============================================================================
#
# Submits torch-fidelity evaluation jobs to Modal cloud.
#
# Usage:
#   ./scripts/evaluate_modal_torch_fidelity.sh \
#       --method ddpm \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt
#
# =============================================================================

set -e

# Defaults optimized for KID evaluation
METHOD="ddpm" # (right now you only have ddpm but you will be implementing more methods as hw progresses)
CHECKPOINT="YOUR_PATH"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo ""
    echo "Usage: $0 --method ddpm --checkpoint <path> [options]"
    echo ""
    echo "Options:"
    echo "  --metrics <fid,kid,is>    Metrics to compute (default: kid)"
    echo "  --num-samples <N>         Number of samples (default: 1000)"
    echo "  --batch-size <N>          Batch size (default: 256)"
    echo "  --num-steps <N>           Sampling steps (default: 1000)"
    exit 1
fi

echo "=========================================="
echo "Modal Torch-Fidelity Evaluation"
echo "=========================================="
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Num steps: $NUM_STEPS"
echo "=========================================="
echo ""
echo "Submitting to Modal..."
echo ""

# Build Modal command
MODAL_CMD="modal run modal_app.py::main --action evaluate_torch_fidelity \
    --method $METHOD \
    --checkpoint $CHECKPOINT \
    --metrics $METRICS \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS"

# Run Modal command
eval $MODAL_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
