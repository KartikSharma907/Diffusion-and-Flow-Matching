#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script (Num-steps Ablation)
# =============================================================================
#
# Runs sampling + torch-fidelity metrics for multiple num_steps values:
#   1, 2, 3, 4, 5, 10, 20, 50, 100, 1000
#
# Usage:
#   ./scripts/evaluate_with_fidelity_steps_ablations.sh \
#       --checkpoint /path/to/checkpoint.pt \
#       --method flow_map_matching \
#       --dataset-path data/celeba-subset/train/images \
#       --metrics kid \
#       --num-samples 1000 \
#       --batch-size 256
#
# Optional:
#   --gpu 3
#   --steps "1 2 3 4 5 10 20 50 100 1000"
#
# =============================================================================

set -euo pipefail

# Defaults
METHOD="flow_matching"
CHECKPOINT="/scr/kartiksh/Diffusion_flow_matching/logs/flowmatching/flow_matching_final.pt"
#"/scr/kartiksh/Diffusion_flow_matching/logs/flow_map_matching_psc_run/flow_map_matching_final.pt"
DATASET_PATH="data/celeba-subset/train/images"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
GPU="3"

# Required ablation steps (can be overridden with --steps)
STEPS_LIST="1 2 3 4 5 10 20 50 100 1000"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
    --method)       METHOD="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --metrics)      METRICS="$2"; shift 2 ;;
    --num-samples)  NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --gpu)          GPU="$2"; shift 2 ;;
    --steps)        STEPS_LIST="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required"
  exit 1
fi

# Set base output directories based on checkpoint location
CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
ABLATION_ROOT="${CHECKPOINT_DIR}/${METHOD}_steps_ablation"
mkdir -p "$ABLATION_ROOT"

SUMMARY_FILE="${ABLATION_ROOT}/summary_metrics.csv"
echo "num_steps,metrics_file,generated_dir,cache_dir" > "$SUMMARY_FILE"

echo "=========================================="
echo "Torch-Fidelity Evaluation (Steps Ablation)"
echo "=========================================="
echo "Checkpoint:     $CHECKPOINT"
echo "Method:         $METHOD"
echo "Dataset:        $DATASET_PATH"
echo "Metrics:        $METRICS"
echo "Num samples:    $NUM_SAMPLES"
echo "Batch size:     $BATCH_SIZE"
echo "GPU (sampling): $GPU"
echo "Steps list:     $STEPS_LIST"
echo "Outputs root:   $ABLATION_ROOT"
echo "=========================================="

for NUM_STEPS in $STEPS_LIST; do
  RUN_TAG="${NUM_STEPS}steps"
  GENERATED_DIR="${ABLATION_ROOT}/${RUN_TAG}/generated"
  CACHE_DIR="${ABLATION_ROOT}/${RUN_TAG}/cache"

  echo ""
  echo "=========================================="
  echo "Running ablation: num_steps=$NUM_STEPS"
  echo "Generated: $GENERATED_DIR"
  echo "Cache:     $CACHE_DIR"
  echo "=========================================="

  # Step 1: Generate samples
  echo ""
  echo "[1/2] Generating samples (num_steps=$NUM_STEPS)..."
  rm -rf "$GENERATED_DIR"
  mkdir -p "$(dirname "$GENERATED_DIR")"

  SAMPLE_CMD=(python sample.py
    --checkpoint "$CHECKPOINT"
    --method "$METHOD"
    --output_dir "$GENERATED_DIR"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --num_steps "$NUM_STEPS"
  )

  CUDA_VISIBLE_DEVICES="$GPU" "${SAMPLE_CMD[@]}"

  # Step 2: Run fidelity
  echo ""
  echo "[2/2] Computing metrics (num_steps=$NUM_STEPS)..."
  rm -rf "$CACHE_DIR"
  mkdir -p "$CACHE_DIR"

  FIDELITY_CMD=(fidelity
    --gpu $GPU
    --batch-size "$BATCH_SIZE"
    --cache-root "$CACHE_DIR"
    --input1 "$GENERATED_DIR"
    --input2 "$DATASET_PATH"
  )

  [[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD+=(--fid)
  [[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD+=(--kid)
  [[ "$METRICS" == *"is"*  ]] && FIDELITY_CMD+=(--isc)

  METRICS_OUTPUT="${CACHE_DIR}/metrics.txt"
  {
    echo "Metrics for $METHOD with $NUM_STEPS steps:"
    echo "Checkpoint: $CHECKPOINT"
    echo "Generated dir: $GENERATED_DIR"
    echo "Dataset: $DATASET_PATH"
    echo "Command: ${FIDELITY_CMD[*]}"
    echo ""
  } > "$METRICS_OUTPUT"

  "${FIDELITY_CMD[@]}" | tee -a "$METRICS_OUTPUT"

  # Record in summary
  echo "${NUM_STEPS},${METRICS_OUTPUT},${GENERATED_DIR},${CACHE_DIR}" >> "$SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "All ablations complete!"
echo "Summary CSV: $SUMMARY_FILE"
echo "Outputs root: $ABLATION_ROOT"
echo "=========================================="