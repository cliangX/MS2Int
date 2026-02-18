#!/usr/bin/env bash
# Rescore pipeline: Mamba + MS2PIP features (with m-ion) rescoring via mokapot
# Usage: bash spectrum_processing/rescore/run_pipeline.sh <WORKDIR> <CKPT_PATH> [CONDA_ENV]
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <WORKDIR> <CKPT_PATH> [CONDA_ENV]" >&2
  exit 2
fi

WORKDIR="$1"
CKPT_PATH="$2"
CONDA_ENV="${3:-mamba_dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_ABS="${CKPT_PATH}"
if [[ "${CKPT_ABS}" != /* ]]; then
  if [[ -f "${CKPT_ABS}" ]]; then
    CKPT_ABS="$(realpath "${CKPT_ABS}")"
  elif [[ -f "${REPO_ROOT}/${CKPT_ABS}" ]]; then
    CKPT_ABS="$(realpath "${REPO_ROOT}/${CKPT_ABS}")"
  else
    echo "[ERROR] Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
  fi
fi

[[ ! -f "${WORKDIR}/txt/msms.txt" ]] && { echo "[ERROR] ${WORKDIR}/txt/msms.txt not found" >&2; exit 1; }
[[ ! -d "${WORKDIR}/mzml" ]] && { echo "[ERROR] ${WORKDIR}/mzml/ not found" >&2; exit 1; }

cd "${WORKDIR}"

FEATURE_TMP_DIR=""
cleanup_feature_tmp() {
  [[ -n "${FEATURE_TMP_DIR}" && -d "${FEATURE_TMP_DIR}" ]] && rm -rf "${FEATURE_TMP_DIR}"
}
trap cleanup_feature_tmp EXIT

echo "[Step 1/6] Filter msms.txt"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/step01_filter_msms_unmodified_len30.py" \
  -i "txt/msms.txt" \
  -o "rescore/1.msms_filtered_unmodified_lenle30.txt"

echo "[Step 2/6] Generate SpecId TSV"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/step02_make_msms_specid.py" \
  "rescore/1.msms_filtered_unmodified_lenle30.txt" \
  "rescore/msms_specid.tsv"

echo "[Step 3/6] Build rescore_batch1.h5"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/unmodficaiton/run.py" \
  --msms "rescore/1.msms_filtered_unmodified_lenle30.txt" \
  --mzml-dir "mzml" \
  --dataset-name "rescore" \
  --output "rescore/rescore_batch1.h5"

echo "[Step 4/6] MS2Int prediction"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/MS2Int/predict.py" \
  --ckpt "${CKPT_ABS}" \
  --input "rescore/rescore_batch1.h5" \
  --output "rescore/rescore.h5"

FEATURE_TMP_DIR="$(mktemp -d "rescore/.features_tmp.XXXXXX")"
MS2PIP_TSV="${FEATURE_TMP_DIR}/ms2pip_features_m.tsv"

echo "[Step 5/6] Compute MS2PIP features"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/step03_calc_ms2pip_features_m.py" \
  --h5_path "rescore/rescore.h5" \
  --tsv_path "rescore/msms_specid.tsv" \
  --output "${MS2PIP_TSV}"

echo "[Step 6/6] Mokapot rescoring"
env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES="" \
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/step04_rescore_mamba_ms2pip_m.py" \
  --msms_path "rescore/1.msms_filtered_unmodified_lenle30.txt" \
  --tsv_path "${MS2PIP_TSV}" \
  --rng 42 --folds 2 --max_workers 2 \
  -v

echo ""
echo "Pipeline complete. Results in ${WORKDIR}/rescore/"
