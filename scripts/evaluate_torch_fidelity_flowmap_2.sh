#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script (Num-steps Ablation)
# =============================================================================
#
# Runs sampling + torch-fidelity metrics for multiple num_steps values and
# dumps a summary Excel file (ablation_results.xlsx) with:
#   KID mean + std, FID, Precision, Recall
#
# Usage:
#   ./scripts/evaluate_torch_fidelity_flowmap_2.sh \
#       --checkpoint /path/to/checkpoint.pt \
#       --method flow_map_matching \
#       --dataset-path data/celeba-subset/train/images \
#       --num-samples 1000 \
#       --batch-size 256
#
# Optional:
#   --gpu 3
#   --steps "1 2 3 4 5 10 20 50 100 1000"
#   --sampler heun|euler
#   --schedule karras|uniform
#   --karras-rho 7.0
#   --metrics "kid fid prc"   (any subset; default: kid fid prc)
#
# =============================================================================

set -euo pipefail

# Capture the Python interpreter from the caller's environment so that an
# activated virtualenv / conda env is respected by the subshell.
PYTHON="$(command -v python)"
# Derive the fidelity binary from the same bin/ as python (same env).
FIDELITY="$(dirname "$PYTHON")/fidelity"
if [[ ! -x "$FIDELITY" ]]; then
  echo "ERROR: 'fidelity' not found at $FIDELITY"
  echo "  Make sure your virtualenv/conda env is activated and torch-fidelity is installed:"
  echo "  pip install torch-fidelity"
  exit 1
fi

# Defaults
METHOD="flow_map_matching"
CHECKPOINT="/scr/kartiksh/Diffusion_flow_matching/logs/flow_map_matching_20260216_082335/checkpoints/flow_map_matching_0135000.pt"
#"/scr/kartiksh/Diffusion_flow_matching/logs/flow_map_matching_psc_run/flow_map_matching_final.pt"
DATASET_PATH="data/celeba-subset/train/images"
METRICS="kid fid"   # kid, fid use InceptionV3 (one pass); add "prc" for P/R (costs a 2nd VGG16 pass)
NUM_SAMPLES=1000
BATCH_SIZE=256
GPU="3"
SAMPLER="heun"     # heun (midpoint-RK2, 2 NFE/step) or euler
SCHEDULE="uniform" # uniform (recommended) or karras (only useful at N>~10 with small rho)
KARRAS_RHO=3.0     # exponent for karras schedule; avoid rho>4 at low N

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
    --sampler)      SAMPLER="$2"; shift 2 ;;
    --schedule)     SCHEDULE="$2"; shift 2 ;;
    --karras-rho)   KARRAS_RHO="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required"
  exit 1
fi

# Set base output directories based on checkpoint location
CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
ABLATION_ROOT="${CHECKPOINT_DIR}/${METHOD}_steps_heun_karras_rho${KARRAS_RHO}"
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
echo "Sampler:        $SAMPLER"
echo "Schedule:       $SCHEDULE (rho=$KARRAS_RHO)"
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

  # --- Step 1: Generate samples ---
  echo ""
  echo "[1/2] Generating samples (num_steps=$NUM_STEPS)..."
  rm -rf "$GENERATED_DIR"
  mkdir -p "$(dirname "$GENERATED_DIR")"

  SAMPLE_CMD=("$PYTHON" sample.py
    --checkpoint "$CHECKPOINT"
    --method "$METHOD"
    --output_dir "$GENERATED_DIR"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --num_steps "$NUM_STEPS"
    --sampler "$SAMPLER"
    --schedule "$SCHEDULE"
    --karras_rho "$KARRAS_RHO"
  )

  CUDA_VISIBLE_DEVICES="$GPU" "${SAMPLE_CMD[@]}"

  # --- Step 2: Run fidelity ---
  echo ""
  echo "[2/2] Computing metrics (num_steps=$NUM_STEPS)..."
  rm -rf "$CACHE_DIR"
  mkdir -p "$CACHE_DIR"

  FIDELITY_CMD=("$FIDELITY"
    --gpu "$GPU"
    --batch-size "$BATCH_SIZE"
    --cache-root "$CACHE_DIR"
    --input1 "$GENERATED_DIR"
    --input2 "$DATASET_PATH"
  )

  [[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD+=(--fid)
  [[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD+=(--kid)
  [[ "$METRICS" == *"is"*  ]] && FIDELITY_CMD+=(--isc)
  [[ "$METRICS" == *"prc"* ]] && FIDELITY_CMD+=(--prc)

  METRICS_OUTPUT="${CACHE_DIR}/metrics.txt"
  {
    echo "Metrics for $METHOD with $NUM_STEPS steps:"
    echo "Checkpoint: $CHECKPOINT"
    echo "Generated dir: $GENERATED_DIR"
    echo "Dataset: $DATASET_PATH"
    echo "Sampler: $SAMPLER | Schedule: $SCHEDULE (rho=$KARRAS_RHO)"
    echo "Command: ${FIDELITY_CMD[*]}"
    echo ""
  } > "$METRICS_OUTPUT"

  "${FIDELITY_CMD[@]}" | tee -a "$METRICS_OUTPUT"

  # Record in summary CSV
  echo "${NUM_STEPS},${METRICS_OUTPUT},${GENERATED_DIR},${CACHE_DIR}" >> "$SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "All ablations complete!"
echo "Summary CSV: $SUMMARY_FILE"
echo "Outputs root: $ABLATION_ROOT"
echo "=========================================="

# =============================================================================
# Build Excel summary: KID mean/std, FID, Precision, Recall
# =============================================================================
echo ""
echo "Building Excel summary..."

"$PYTHON" - "$ABLATION_ROOT" "$METHOD" "$SAMPLER" "$SCHEDULE" "$KARRAS_RHO" <<'PYEOF'
import sys
import os
import re
import glob

ablation_root = sys.argv[1]
method        = sys.argv[2]
sampler       = sys.argv[3]
schedule      = sys.argv[4]
karras_rho    = sys.argv[5]

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Warning] pandas not found – falling back to plain CSV output.")

try:
    import openpyxl  # noqa: F401
    EXCEL_ENGINE = "openpyxl"
except ImportError:
    try:
        import xlsxwriter  # noqa: F401
        EXCEL_ENGINE = "xlsxwriter"
    except ImportError:
        EXCEL_ENGINE = None
        print("[Warning] Neither openpyxl nor xlsxwriter found – will save CSV only.")


# ---------------------------------------------------------------------------
# Regex patterns for torch-fidelity metric output lines
# ---------------------------------------------------------------------------
METRIC_PATTERNS = {
    "KID Mean":   r"Kernel Inception Distance Mean:\s*([\d.eE+\-]+)",
    "KID Std":    r"Kernel Inception Distance Std:\s*([\d.eE+\-]+)",
    "FID":        r"Frechet Inception Distance:\s*([\d.eE+\-]+)",
    "Precision":  r"Precision:\s*([\d.eE+\-]+)",
    "Recall":     r"Recall:\s*([\d.eE+\-]+)",
    "IS Mean":    r"Inception Score Mean:\s*([\d.eE+\-]+)",
    "IS Std":     r"Inception Score Std:\s*([\d.eE+\-]+)",
}


def parse_metrics(metrics_file: str) -> dict:
    """Extract numeric metric values from a torch-fidelity metrics.txt."""
    result = {}
    if not os.path.exists(metrics_file):
        return result
    with open(metrics_file) as f:
        text = f.read()
    for col, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            result[col] = float(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Discover all *steps run directories, sorted numerically
# ---------------------------------------------------------------------------
def _step_count(path: str) -> int:
    digits = "".join(filter(str.isdigit, os.path.basename(path)))
    return int(digits) if digits else 0

run_dirs = sorted(
    glob.glob(os.path.join(ablation_root, "*steps")),
    key=_step_count,
)

rows = []
for run_dir in run_dirs:
    num_steps = _step_count(run_dir)
    metrics_file = os.path.join(run_dir, "cache", "metrics.txt")
    metrics = parse_metrics(metrics_file)

    row = {
        "num_steps":  num_steps,
        "sampler":    sampler,
        "schedule":   schedule,
        "karras_rho": float(karras_rho),
    }
    row.update(metrics)
    rows.append(row)

if not rows:
    print("No ablation results found – nothing to write.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
excel_path = os.path.join(ablation_root, "ablation_results.xlsx")
csv_path   = os.path.join(ablation_root, "ablation_results.csv")

if not HAS_PANDAS:
    # Manual CSV fallback
    import csv
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV summary saved to: {csv_path}")
    sys.exit(0)

import pandas as pd
df = pd.DataFrame(rows).sort_values("num_steps").reset_index(drop=True)

# Always write CSV as a reliable backup
df.to_csv(csv_path, index=False)
print(f"CSV backup saved to: {csv_path}")

# ---- Excel -----------------------------------------------------------------
if EXCEL_ENGINE is None:
    print("Skipping Excel (no suitable engine installed).")
    print("  Install with:  pip install openpyxl")
else:
    with pd.ExcelWriter(excel_path, engine=EXCEL_ENGINE) as writer:
        df.to_excel(writer, index=False, sheet_name="Ablation")

        # ---- Formatting (openpyxl only) ------------------------------------
        if EXCEL_ENGINE == "openpyxl":
            from openpyxl.styles import (
                Font, PatternFill, Alignment, Border, Side,
            )

            ws = writer.sheets["Ablation"]

            # Header style
            hdr_fill  = PatternFill("solid", fgColor="1F497D")
            hdr_font  = Font(color="FFFFFF", bold=True, size=11)
            hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # Alternating row fill
            alt_fill = PatternFill("solid", fgColor="DCE6F1")

            thin   = Side(border_style="thin", color="B8CCE4")
            border = Border(left=thin, right=thin, top=thin, bottom=thin)

            n_rows = len(df) + 1  # +1 for header

            for cell in next(ws.iter_rows(min_row=1, max_row=1)):
                cell.fill      = hdr_fill
                cell.font      = hdr_font
                cell.alignment = hdr_align
                cell.border    = border

            for row_idx, row_cells in enumerate(
                ws.iter_rows(min_row=2, max_row=n_rows), start=2
            ):
                fill = alt_fill if row_idx % 2 == 0 else PatternFill()
                for cell in row_cells:
                    cell.fill      = fill
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border    = border
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"

            # Auto-size columns
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) for c in col_cells if c.value is not None),
                    default=8,
                )
                ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4

            ws.freeze_panes = "A2"

            # Plain sheet for easy copy-paste
            ws2 = writer.book.create_sheet("Plain")
            ws2.append(list(df.columns))
            for _, r in df.iterrows():
                ws2.append(list(r))

    print(f"Excel summary saved to: {excel_path}")

# ---------------------------------------------------------------------------
# Print table to stdout
# ---------------------------------------------------------------------------
print("")
print(df.to_string(index=False))
PYEOF

echo ""
echo "=========================================="
echo "Excel/CSV summary written to:"
echo "  ${ABLATION_ROOT}/ablation_results.xlsx"
echo "  ${ABLATION_ROOT}/ablation_results.csv"
echo "=========================================="
