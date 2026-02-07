#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 一键运行：Mamba + 全特征（含 m 离子）重打分
#
# 用法：
#   bash spectrum_processing/rescore/run_mamba_all_features.sh "<WORKDIR>" "<CKPT_PATH>" ["CONDA_ENV"]
#
# 约定：
#   WORKDIR/
#     ├── txt/msms.txt
#     └── mzml/*.mzML
#
# 输出：
#   WORKDIR/rescore/ 下生成中间与最终结果

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 \"<WORKDIR>\" \"<CKPT_PATH>\" [\"CONDA_ENV\"]" >&2
  exit 2
fi

WORKDIR="$1"
CKPT_PATH="$2"
CONDA_ENV="${3:-mamba_dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

step() {
  local title="$1"; shift
  echo
  echo "------------------------------------------------------------"
  echo "[RUN] ${title}"
  echo "------------------------------------------------------------"
  "$@"
}

if [[ ! -d "${WORKDIR}/txt" || ! -f "${WORKDIR}/txt/msms.txt" ]]; then
  echo "[ERROR] 未找到 \"${WORKDIR}/txt/msms.txt\"" >&2
  exit 1
fi
if [[ ! -d "${WORKDIR}/mzml" ]]; then
  echo "[ERROR] 未找到 \"${WORKDIR}/mzml\" 目录" >&2
  exit 1
fi

mkdir -p "${WORKDIR}/rescore/logs"

cd "${WORKDIR}"

step "Step 1/8 过滤 msms.txt（Unmodified + Length<=30 + 去掉含U序列）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/0.filter_msms_30_unmodified.py" \
    -i "txt/msms.txt" \
    -o "rescore/1.msms_filtered_unmodified_lenle30.txt"

step "Step 2/8 生成 msms_specid.tsv" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/1.make_msms_specid.py" \
    "rescore/1.msms_filtered_unmodified_lenle30.txt" \
    "rescore/msms_specid.tsv"

step "Step 3/8 制作 rescore_batch1.h5（真实谱图 + train_data）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/unmodficaiton/run.py" \
    --msms "rescore/1.msms_filtered_unmodified_lenle30.txt" \
    --mzml-dir "mzml" \
    --dataset-name "rescore" \
    --output-dir "rescore" \
    --final-h5 "rescore/rescore_batch1.h5"

step "Step 4/8 MS2Int 推理（写入 Intpredict -> rescore.h5）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/MS2Int/predict.py" \
    --ckpt "${CKPT_PATH}" \
    --input "rescore/rescore_batch1.h5" \
    --output "rescore/rescore.h5"

step "Step 5/8 为 rescore.h5 写入 SpecId" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/5.add_SpecId_2_h5.py" \
    --h5_path "rescore/rescore.h5"

step "Step 6/8 计算 MS2PIP 特征（含 m 离子）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/6m.calculator_ms2pip_feature_m.py" \
    --h5_path "rescore/rescore.h5" \
    --tsv_path "rescore/msms_specid.tsv" \
    --output "rescore/msms_specid_with_MS2PIP_m.tsv"

step "Step 7/8 计算 OK 特征并生成合并版 TSV（ms2pip+ok，含 m 离子）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/7m.calculator_ok_feature_m.py" \
    --h5_path "rescore/rescore.h5" \
    --tsv_path "rescore/msms_specid_with_MS2PIP_m.tsv" \
    --output "rescore/msms_specid_with_ms2pip_ok_m.tsv"

step "Step 8/8 mokapot 重打分（Mamba + 全特征）" \
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/spectrum_processing/rescore/10bm.rescore_mamba_with_ms2pip_ok_m.py" \
    --msms_path "rescore/1.msms_filtered_unmodified_lenle30.txt" \
    --tsv_path "rescore/msms_specid_with_ms2pip_ok_m.tsv" \
    --rng 42 --folds 2 --max_workers 2 \
    --log_path "rescore/logs/rescore.log" \
    -v

echo
echo "[OK] 全部完成。主要结果目录：\"${WORKDIR}/rescore/rescore_mamba_ok_m/\""

