#!/bin/bash
# MambaFLR: 磷酸化位点定位质量控制流水线
# 用法: bash run_pipeline.sh PROJECT_ROOT
# 示例: bash /mnt/data_nas/lcy/project_MS2predict/5.tools/mambaflr/mambaflr/run_pipeline.sh /mnt/data_nas/lcy/project_MS2predict/5.tools/PTM_benchmark/3.data.flr.eval/

set -e
export HDF5_USE_FILE_LOCKING=FALSE

# ============ 颜色与日志 ============
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
error() { echo -e "${RED}[ERROR]${NC} $1"; }
log() { echo -e "${CYAN}$1${NC}"; }

# ============ 参数校验 ============
if [ $# -lt 1 ]; then
  echo "用法: $0 PROJECT_ROOT"
  exit 1
fi

PROJECT_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============ 配置参数 ============
MODEL_CKPT="${MODEL_CKPT:-}"
if [ -z "$MODEL_CKPT" ]; then
  error "未设置模型权重路径。请先设置环境变量: MODEL_CKPT=/path/to/model.pth"
  exit 1
fi
BATCH_SIZE=1024
TARGET_FLR=0.01

# ============ 目录结构 ============
OUT_DIR="${PROJECT_ROOT}/mambaflr"
RESCORE_DIR="${OUT_DIR}/rescore"
PROG_DIR="${OUT_DIR}/.progress"

# ============ 进度管理 ============
step_done() { [ -f "${PROG_DIR}/$1" ]; }
mark_done() {
  mkdir -p "${PROG_DIR}"
  echo "[$(date '+%F %T')] $2" > "${PROG_DIR}/$1"
  log "  ✓ $2"
}

# ============ 格式化输出配置 ============
print_config() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║                   MambaFLR Pipeline                        ║"
  echo "╠════════════════════════════════════════════════════════════╣"
  printf "║ %-20s %-37s ║\n" "PROJECT_ROOT:" "$(basename "$PROJECT_ROOT")"
  printf "║ %-20s %-37s ║\n" "MODEL_CKPT:" "$(basename "$MODEL_CKPT")"
  printf "║ %-20s %-37s ║\n" "BATCH_SIZE:" "$BATCH_SIZE"
  printf "║ %-20s %-37s ║\n" "TARGET_FLR:" "$TARGET_FLR"
  printf "║ %-20s %-37s ║\n" "OUTPUT_DIR:" "mambaflr/"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo ""
}

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT_DIR" "$RESCORE_DIR" "$PROG_DIR"
print_config

# ============ Step 1: 生成 Target/Decoy 列表 ============
log "[Step 1/8] 生成 Target/Decoy 列表"
if step_done "step1.done"; then
  log "  → 跳过（已完成）"
else
  python "${SCRIPT_DIR}/step1_generate_TD_list.py" \
    --inputfile "txt/msms.txt" \
    --outputfile "${OUT_DIR}/step1_TD_list.csv" \
    --quiet
  mark_done "step1.done" "Target/Decoy 列表生成完成"
fi

# ============ Step 2: 创建 TD DataFrame ============
log "[Step 2/8] 创建 TD DataFrame"
if step_done "step2.done"; then
  log "  → 跳过（已完成）"
else
  python "${SCRIPT_DIR}/step2_create_TD_df.py" \
    --inputfile "${OUT_DIR}/step1_TD_list.csv" \
    --outputfile "${OUT_DIR}/step2_TD_df.csv" \
    --quiet
  mark_done "step2.done" "TD DataFrame 创建完成"
fi

# ============ Step 3: 构建参考谱图 H5 ============
log "[Step 3/8] 构建参考谱图 H5"
if step_done "step3.done"; then
  log "  → 跳过（已完成）"
else
  [ -f "${RESCORE_DIR}/step3_ref_spectra.h5" ] && rm -f "${RESCORE_DIR}/step3_ref_spectra.h5"
  
  if ! python -c 'import pyopenms' 2>/dev/null; then
    error "未安装 pyopenms，请先安装: pip install pyopenms"
    exit 1
  fi
  
  python "${REPO_ROOT}/spectrum_processing/step3_build_ref_h5.py" \
    --target_decoy_csv "${OUT_DIR}/step1_TD_list.csv" \
    --msms "txt/msms.txt" \
    --mzml-dir "mzml" \
    --output "${RESCORE_DIR}/step3_ref_spectra.h5" \
    --quiet
  mark_done "step3.done" "参考谱图 H5 构建完成"
fi

# ============ Step 4: 转换为 Mamba 输入 H5 ============
log "[Step 4/8] 转换为 Mamba 输入 H5"
if step_done "step4.done"; then
  log "  → 跳过（已完成）"
else
  python "${REPO_ROOT}/data_processing/step4_convert_to_mamba_h5.py" \
    --input "${OUT_DIR}/step2_TD_df.csv" \
    --output "${OUT_DIR}/step4_mamba_input.h5" \
    --collision_energy 35 \
    --fragmentation CID \
    --ref_h5 "${RESCORE_DIR}/step3_ref_spectra.h5" \
    --quiet
  mark_done "step4.done" "Mamba 输入 H5 转换完成"
fi

# ============ Step 5: Mamba 谱图预测 ============
log "[Step 5/8] Mamba 谱图预测"
if step_done "step5.done"; then
  log "  → 跳过（已完成）"
else
  MAMBA_H5="${OUT_DIR}/step4_mamba_input.h5"
  [ ! -f "$MAMBA_H5" ] && { error "找不到: $MAMBA_H5"; exit 1; }
  
  # 强制设置 Fragmentation=HCD, collision_energy=30
  python - "$MAMBA_H5" <<'PY'
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "r+") as f:
    n = f["Fragmentation"].shape[0]
    f["Fragmentation"][:] = np.array([b"HCD"] * n, dtype=f["Fragmentation"].dtype)
    f["collision_energy"][:] = np.full(n, 30.0, dtype=f["collision_energy"].dtype)
PY
  
  MODEL_CKPT="$MODEL_CKPT" bash "${SCRIPT_DIR}/step5_ms2int_predict.sh" \
    "$MAMBA_H5" "$MAMBA_H5" "$BATCH_SIZE"
  mark_done "step5.done" "Mamba 预测完成"
fi

# ============ Step 6: 计算谱图相似度 ============
log "[Step 6/8] 计算谱图相似度 (Cosine)"
if step_done "step6.done"; then
  log "  → 跳过（已完成）"
else
  python "${SCRIPT_DIR}/step6_compute_Cosine.py" \
    --pred_h5 "${OUT_DIR}/step4_mamba_input.h5" \
    --ref_h5 "${RESCORE_DIR}/step3_ref_spectra.h5" \
    --pred_key Intpredict \
    --true_key train_data \
    --n 31 --mode flatten --align index \
    --template_csv "${OUT_DIR}/step2_TD_df.csv"
  
  cp "${OUT_DIR}/step2_TD_df.csv" "${OUT_DIR}/step6_df_score.csv"
  mark_done "step6.done" "谱图相似度计算完成"
fi

# ============ Step 7: 计算 FLR 曲线 ============
log "[Step 7/8] 计算 FLR 曲线"
if step_done "step7.done"; then
  log "  → 跳过（已完成）"
else
  python "${SCRIPT_DIR}/step7_compute_flr.py" \
    --modelresultfile "${OUT_DIR}/step6_df_score.csv" \
    --sequencefile "${OUT_DIR}/step1_TD_list.csv" \
    --outputfile "${OUT_DIR}/step7_flr_curve.csv" \
    --psm_outputfile "${OUT_DIR}/step7_unique_psm.csv"
  
  # 汇报 FLR 结果
  FLR_CSV="${OUT_DIR}/step7_flr_curve.csv"
  if [ -f "$FLR_CSV" ]; then
    TOTAL_PSM=$(awk -F, 'NR==2 {print $3}' "$FLR_CSV" | cut -d. -f1)
    FLR_MIN=$(awk -F, 'END {print $2}' "$FLR_CSV")
    FLR_MAX=$(awk -F, 'NR==2 {print $2}' "$FLR_CSV")
    log "  FLR 范围: ${FLR_MIN} - ${FLR_MAX}"
    for T in 0.01 0.02 0.05; do
      PSM=$(awk -F, -v t="$T" 'NR>1 && $2<=t {print $3; exit}' "$FLR_CSV" | cut -d. -f1)
      [ -z "$PSM" ] && PSM=0
      PCT=$(echo "$T * 100" | bc | cut -d. -f1)
      log "  FLR ≤ ${PCT}%: ${PSM}/${TOTAL_PSM}"
    done
  fi
  mark_done "step7.done" "FLR 曲线计算完成"
fi

# ============ Step 8: 导出磷酸化位点 ============
log "[Step 8/8] 导出磷酸化位点"
STY_FILE="txt/Phospho (STY)Sites.txt"

if [ ! -f "$STY_FILE" ]; then
  log "  ⚠ 未找到 $STY_FILE，跳过位点导出"
else
  if step_done "step8.done"; then
    log "  → 跳过（已完成）"
  else
    FLR_CSV="${OUT_DIR}/step7_flr_curve.csv"
    DELTA_CUTOFF=$(awk -F, -v t="$TARGET_FLR" 'NR>1 && $2<=t {print $1; exit}' "$FLR_CSV")
    [ -z "$DELTA_CUTOFF" ] && DELTA_CUTOFF=$(awk -F, 'NR==2 {print $1}' "$FLR_CSV")
    
    log "  → Delta cutoff (FLR≤${TARGET_FLR}): $DELTA_CUTOFF"
    
    python "${SCRIPT_DIR}/step8_export_phosphosites.py" \
      --modelresultfile "${OUT_DIR}/step6_df_score.csv" \
      --sequencefile "${OUT_DIR}/step1_TD_list.csv" \
      --inputfile1 "txt/msms.txt" \
      --inputfile2 "$STY_FILE" \
      --cutoff "$DELTA_CUTOFF" \
      --outputresult "${OUT_DIR}/step8_phosphosites.csv"
    mark_done "step8.done" "磷酸化位点导出完成"
  fi
fi

# ============ 完成总结 ============
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline 完成                           ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║ 输出文件:                                                  ║"
printf "║   %-56s ║\n" "mambaflr/step1_TD_list.csv"
printf "║   %-56s ║\n" "mambaflr/step6_df_score.csv"
printf "║   %-56s ║\n" "mambaflr/step7_flr_curve.csv"
printf "║   %-56s ║\n" "mambaflr/step8_phosphosites.csv"
echo "╚════════════════════════════════════════════════════════════╝"
