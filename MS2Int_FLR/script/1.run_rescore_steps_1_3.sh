#!/usr/bin/env bash
# Steps 1-3 only. Only the working directory is a parameter; others are fixed.
# Usage:
#     
#    bash /mnt/data_nas/lcy/project_MS2predict/5.tools/DeepFLR/mamba_DeepFLR/script/1.run_rescore_steps_1_3.sh /mnt/data_nas/lcy/project_MS2predict/1.data/independent_test/PTM/PXD000138/Finetune_data2

set -euo pipefail

# 关闭 HDF5 文件锁，避免在 NFS / NAS 上出现 unable to lock file 错误
export HDF5_USE_FILE_LOCKING=FALSE

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workdir>" >&2
  exit 2
fi

WORKDIR="$1"
# 使用当前仓库下的 extract_real_spectrums 工具脚本
# 本脚本位于:  /.../DeepFLR/mamba_DeepFLR/script
# REPO_ROOT 即: /.../DeepFLR/mamba_DeepFLR
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_ROOT="$REPO_ROOT/extract_real_spectrums"

# -------- unified log header --------
HR='------------------------------------------------------------'
step_start() {
  # $1: step number  $2: title
  echo
  echo "$HR"
  echo "[START] Step $1 - $2"
  echo "$HR"
}

cd "$WORKDIR"
mkdir -p rescore progress

PROGRESS_DIR="$WORKDIR/progress"

# 01) Filter msms (length<=30, Phospho)
step_start 1 "过滤 msms (Phospho (STY), Length<=30)"
STEP_DONE="$PROGRESS_DIR/01_filter_msms.done"
if [[ -f "$STEP_DONE" ]]; then
  echo "[Skip] Step 1 已完成，跳过。标记: $(basename "$STEP_DONE")"
else
  # Note: ensure conda shell is initialized if needed
  # conda activate mamba || true
  python "$TOOLS_ROOT/0.filter_msms_30_phospho.py" \
    -i txt/msms.txt \
    -o rescore/1.msms_filtered_phospho_sty_lenle30.txt
  touch "$STEP_DONE"
  echo "[Done] Step 1 完成，已写入: $(basename "$STEP_DONE")"
fi

# 02) Make msms_specid file
step_start 2 "生成 msms_specid"
STEP_DONE="$PROGRESS_DIR/02_msms_specid.done"
if [[ -f "$STEP_DONE" ]]; then
  echo "[Skip] Step 2 已完成，跳过。标记: $(basename "$STEP_DONE")"
else
  python "$TOOLS_ROOT/1.make_msms_specid.py" \
    rescore/1.msms_filtered_phospho_sty_lenle30.txt \
    rescore/msms_specid.tsv
  touch "$STEP_DONE"
  echo "[Done] Step 2 完成，已写入: $(basename "$STEP_DONE")"
fi

# 03) Extract real spectrums
# (originally suggested to run on host 75; adjust env as needed)
step_start 3 "提取真实谱图 (extract_real_spectrums)"
STEP_DONE="$PROGRESS_DIR/03_extract_real_spectrums.done"
if [[ -f "$STEP_DONE" ]]; then
  echo "[Skip] Step 3 已完成，跳过。标记: $(basename "$STEP_DONE")"
else
  python "$TOOLS_ROOT/run.py" \
    --msms rescore/1.msms_filtered_phospho_sty_lenle30.txt \
    --mzml-dir mzml \
    --num-workers 32 \
    --batch-size 400 \
    --output-dir rescore
  touch "$STEP_DONE"
  echo "[Done] Step 3 完成，已写入: $(basename "$STEP_DONE")"
fi

# 标准化文件名：确保文件名为 origin_data.h5
if [[ -f rescore/origin_data.h5 ]]; then
  echo "[Skip] rescore/origin_data.h5 已存在，跳过重命名步骤"
else
  shopt -s nullglob
  candidates=(rescore/*_batch1.h5)
  found=false
  for candidate in "${candidates[@]}"; do
    if [[ "$(basename "$candidate")" != "origin_data.h5" ]]; then
      echo "将 $candidate 重命名为 rescore/origin_data.h5 以兼容后续步骤"
      mv -f "$candidate" rescore/origin_data.h5
      found=true
      break
    fi
  done
  if [[ "$found" == "false" ]]; then
    echo "未找到 batch1 产物 (rescore/*_batch1.h5)，无法继续" >&2
    exit 1
  fi
fi

echo "Steps 1-3 completed."
