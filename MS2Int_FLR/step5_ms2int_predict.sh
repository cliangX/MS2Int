#!/bin/bash
# Run MS2Int prediction: input.h5 -> Intpredict dataset
# Usage: bash step5_ms2int_predict.sh input.h5 [output.h5] [batch_size]
set -e

if [ $# -lt 1 ]; then
    echo "[ERROR] Usage: $0 input.h5 [output.h5] [batch_size]"
    exit 1
fi

INPUT_H5="$1"
OUTPUT_H5="${2:-$INPUT_H5}"
BATCH_SIZE="${3:-1024}"

MODEL_CKPT="${MODEL_CKPT:-}"
[ -z "$MODEL_CKPT" ] && { echo "[ERROR] MODEL_CKPT not set"; exit 1; }
[ ! -f "$INPUT_H5" ] && { echo "[ERROR] Input not found: $INPUT_H5"; exit 1; }
[ ! -f "$MODEL_CKPT" ] && { echo "[ERROR] Checkpoint not found: $MODEL_CKPT"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREDICT_PY="${REPO_ROOT}/MS2Int/predict.py"
[ ! -f "$PREDICT_PY" ] && { echo "[ERROR] predict.py not found: $PREDICT_PY"; exit 1; }

echo "[Step5] MS2Int predict: $(basename "$INPUT_H5")"

export HDF5_USE_FILE_LOCKING=FALSE
${PY_BIN:-python} "$PREDICT_PY" \
  --checkpoint_path "$MODEL_CKPT" \
  --input_path "$INPUT_H5" \
  --output_path "$OUTPUT_H5"
