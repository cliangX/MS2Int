#!/bin/bash
# Step 5: MS2Int 谱图预测
# 用法: bash step5_ms2int_predict.sh input.h5 [output.h5] [batch_size]
# 说明: 调用 ../MS2Int/predict.py 进行推理，输出写入 Intpredict 数据集

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}$1${NC}"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [ $# -lt 1 ]; then
    error "Usage: $0 input.h5 [output.h5] [batch_size]"
    exit 1
fi

INPUT_H5="$1"
OUTPUT_H5="${2:-$INPUT_H5}"
BATCH_SIZE="${3:-1024}"      # 目前只用于日志输出

MODEL_CKPT="${MODEL_CKPT:-}"
if [ -z "$MODEL_CKPT" ]; then
    error "未设置模型权重路径。请先设置环境变量: MODEL_CKPT=/path/to/model.pth"
    exit 1
fi

if [ ! -f "$INPUT_H5" ]; then
    error "Input file not found: $INPUT_H5"
    exit 1
fi
if [ ! -f "$MODEL_CKPT" ]; then
    error "Model checkpoint not found: $MODEL_CKPT"
    exit 1
fi

log "[Step5] MS2Int 预测: $(basename "$INPUT_H5")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREDICT_PY="${REPO_ROOT}/MS2Int/predict.py"

if [ ! -f "$PREDICT_PY" ]; then
    error "找不到本地预测脚本: $PREDICT_PY"
    exit 1
fi

PY_BIN="${PY_BIN:-python}"
# 避免网络盘 HDF5 锁导致阻塞
export HDF5_USE_FILE_LOCKING=FALSE

$PY_BIN "$PREDICT_PY" \
  --checkpoint_path "$MODEL_CKPT" \
  --input_path "$INPUT_H5" \
  --output_path "$OUTPUT_H5"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    error "❌ 本地 Mamba 预测脚本执行失败，退出码: $STATUS"
    exit $STATUS
fi
