#!/bin/bash
# MS2Int-FLR: PTM site localization quality control pipeline
# Usage: bash run_pipeline.sh PROJECT_ROOT CKPT_PATH
# Example: bash MS2Int_FLR/run_pipeline.sh data/MS2Int_flr ptm_finetune/checkpoints/model_epoch_14_val_loss_0.3056_0627_123641.pth
set -e
export HDF5_USE_FILE_LOCKING=FALSE
PY_BIN="${PY_BIN:-python}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 PROJECT_ROOT CKPT_PATH"; exit 1
fi

CALL_DIR="$(pwd -P)"
PROJECT_ROOT_ARG="$1"
MODEL_CKPT_ARG="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "$PROJECT_ROOT_ARG" in
  /*) PROJECT_ROOT="$PROJECT_ROOT_ARG" ;;
  *) PROJECT_ROOT="${CALL_DIR}/${PROJECT_ROOT_ARG}" ;;
esac
case "$MODEL_CKPT_ARG" in
  /*) MODEL_CKPT="$MODEL_CKPT_ARG" ;;
  *) MODEL_CKPT="${CALL_DIR}/${MODEL_CKPT_ARG}" ;;
esac

[ ! -d "$PROJECT_ROOT" ] && { echo "[ERROR] Project root not found: $PROJECT_ROOT"; exit 1; }
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
[ ! -f "$MODEL_CKPT" ] && { echo "[ERROR] Checkpoint not found: $MODEL_CKPT"; exit 1; }
MODEL_CKPT="$(cd "$(dirname "$MODEL_CKPT")" && pwd)/$(basename "$MODEL_CKPT")"
BATCH_SIZE=1024
TARGET_FLR=0.01

OUT_DIR="${PROJECT_ROOT}/output"
RESCORE_DIR="${OUT_DIR}/rescore"
PROG_DIR="${OUT_DIR}/.progress"

step_done() { [ -f "${PROG_DIR}/$1" ]; }
mark_done() {
  mkdir -p "${PROG_DIR}"
  echo "[$(date '+%F %T')] $2" > "${PROG_DIR}/$1"
  echo "  done: $2"
}

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT_DIR" "$RESCORE_DIR" "$PROG_DIR"
echo "PROJECT_ROOT=$(basename "$PROJECT_ROOT")  MODEL=$(basename "$MODEL_CKPT")  FLR=$TARGET_FLR"

# Step 1: Generate target/decoy list
echo "[Step 1/8] Generate TD list"
if step_done "step1.done"; then echo "  skip"; else
  ${PY_BIN} "${SCRIPT_DIR}/step1_generate_TD_list.py" \
    --inputfile "txt/msms.txt" \
    --outputfile "${OUT_DIR}/step1_TD_list.csv" --quiet
  mark_done "step1.done" "TD list"
fi

# Step 2: Create TD DataFrame
echo "[Step 2/8] Create TD DataFrame"
if step_done "step2.done"; then echo "  skip"; else
  ${PY_BIN} "${SCRIPT_DIR}/step2_create_TD_df.py" \
    --inputfile "${OUT_DIR}/step1_TD_list.csv" \
    --outputfile "${OUT_DIR}/step2_TD_df.csv" --quiet
  mark_done "step2.done" "TD DataFrame"
fi

# Step 3: Build reference spectra H5
echo "[Step 3/8] Build reference H5"
if step_done "step3.done"; then echo "  skip"; else
  [ -f "${RESCORE_DIR}/step3_ref_spectra.h5" ] && rm -f "${RESCORE_DIR}/step3_ref_spectra.h5"
  ${PY_BIN} -c 'import pyopenms' 2>/dev/null || { echo "[ERROR] pyopenms not installed"; exit 1; }
  ${PY_BIN} "${SCRIPT_DIR}/step3_build_ref_h5.py" \
    --target_decoy_csv "${OUT_DIR}/step1_TD_list.csv" \
    --msms "txt/msms.txt" --mzml-dir "mzml" \
    --output "${RESCORE_DIR}/step3_ref_spectra.h5" --quiet
  mark_done "step3.done" "Reference H5"
fi

# Step 4: Convert to Mamba input H5
echo "[Step 4/8] Convert to Mamba H5"
if step_done "step4.done"; then echo "  skip"; else
  ${PY_BIN} "${SCRIPT_DIR}/step4_convert_to_mamba_h5.py" \
    --input "${OUT_DIR}/step2_TD_df.csv" \
    --output "${OUT_DIR}/step4_mamba_input.h5" \
    --collision_energy 35 --fragmentation CID \
    --ref_h5 "${RESCORE_DIR}/step3_ref_spectra.h5" --quiet
  mark_done "step4.done" "Mamba H5"
fi

# Step 5: MS2Int prediction
echo "[Step 5/8] MS2Int prediction"
if step_done "step5.done"; then echo "  skip"; else
  MAMBA_H5="${OUT_DIR}/step4_mamba_input.h5"
  [ ! -f "$MAMBA_H5" ] && { echo "[ERROR] Not found: $MAMBA_H5"; exit 1; }

  # Override Fragmentation=HCD, collision_energy=30 for prediction
  ${PY_BIN} - "$MAMBA_H5" <<'PY'
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "r+") as f:
    n = f["Fragmentation"].shape[0]
    f["Fragmentation"][:] = np.array([b"HCD"] * n, dtype=f["Fragmentation"].dtype)
    f["collision_energy"][:] = np.full(n, 30.0, dtype=f["collision_energy"].dtype)
PY

  MODEL_CKPT="$MODEL_CKPT" bash "${SCRIPT_DIR}/step5_ms2int_predict.sh" \
    "$MAMBA_H5" "$MAMBA_H5" "$BATCH_SIZE"
  mark_done "step5.done" "MS2Int prediction"
fi

# Step 6: Compute cosine similarity
echo "[Step 6/8] Compute cosine similarity"
if step_done "step6.done"; then echo "  skip"; else
  ${PY_BIN} "${SCRIPT_DIR}/step6_compute_Cosine.py" \
    --pred_h5 "${OUT_DIR}/step4_mamba_input.h5" \
    --ref_h5 "${RESCORE_DIR}/step3_ref_spectra.h5" \
    --pred_key Intpredict --true_key train_data \
    --n 41 --mode flatten --align index \
    --template_csv "${OUT_DIR}/step2_TD_df.csv"
  cp "${OUT_DIR}/step2_TD_df.csv" "${OUT_DIR}/step6_df_score.csv"
  mark_done "step6.done" "Cosine similarity"
fi

# Step 7: Compute FLR curve
echo "[Step 7/8] Compute FLR curve"
if step_done "step7.done"; then echo "  skip"; else
  ${PY_BIN} "${SCRIPT_DIR}/step7_compute_flr.py" \
    --modelresultfile "${OUT_DIR}/step6_df_score.csv" \
    --sequencefile "${OUT_DIR}/step1_TD_list.csv" \
    --outputfile "${OUT_DIR}/step7_flr_curve.csv" \
    --psm_outputfile "${OUT_DIR}/step7_unique_psm.csv"

  FLR_CSV="${OUT_DIR}/step7_flr_curve.csv"
  if [ -f "$FLR_CSV" ]; then
    TOTAL_PSM=$(awk -F, 'NR==2 {print $3}' "$FLR_CSV" | cut -d. -f1)
    for T in 0.01 0.02 0.05; do
      PSM=$(awk -F, -v t="$T" 'NR>1 && $2<=t {print $3; exit}' "$FLR_CSV" | cut -d. -f1)
      [ -z "$PSM" ] && PSM=0
      PCT=$(echo "$T * 100" | bc | cut -d. -f1)
      echo "  FLR<=${PCT}%: ${PSM}/${TOTAL_PSM}"
    done
  fi
  mark_done "step7.done" "FLR curve"
fi

# Step 8: Export phosphosites
echo "[Step 8/8] Export phosphosites"
STY_FILE="txt/Phospho (STY)Sites.txt"
if [ ! -f "$STY_FILE" ]; then
  echo "  $STY_FILE not found, skipping"
else
  if step_done "step8.done"; then echo "  skip"; else
    FLR_CSV="${OUT_DIR}/step7_flr_curve.csv"
    DELTA_CUTOFF=$(awk -F, -v t="$TARGET_FLR" 'NR>1 && $2<=t {print $1; exit}' "$FLR_CSV")
    [ -z "$DELTA_CUTOFF" ] && DELTA_CUTOFF=$(awk -F, 'NR==2 {print $1}' "$FLR_CSV")
    echo "  Delta cutoff (FLR<=$TARGET_FLR): $DELTA_CUTOFF"

    ${PY_BIN} "${SCRIPT_DIR}/step8_export_phosphosites.py" \
      --modelresultfile "${OUT_DIR}/step6_df_score.csv" \
      --sequencefile "${OUT_DIR}/step1_TD_list.csv" \
      --inputfile1 "txt/msms.txt" \
      --inputfile2 "$STY_FILE" \
      --cutoff "$DELTA_CUTOFF" \
      --outputresult "${OUT_DIR}/step8_phosphosites.csv"
    mark_done "step8.done" "Phosphosite export"
  fi
fi

echo ""
echo "Pipeline complete. Outputs in ${OUT_DIR}/"
