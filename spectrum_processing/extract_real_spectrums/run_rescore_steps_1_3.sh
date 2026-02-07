#!/usr/bin/env bash
# 磷酸化谱图提取流程（步骤 1-3）封装脚本
# 作用：将旧版基于 mamba_rescore 的实现，切换为当前仓库
#       DeepFLR/mamba_DeepFLR/script/1.run_rescore_steps_1_3.sh 中的
#       磷酸化专用谱图提取流程（使用 extract_real_spectrums 下的代码）。
#
# 用法示例：
#   bash /mnt/data_nas/lcy/project_MS2predict/5.tools/mambaflr/mambaflr/extract_real_spectrums/run_rescore_steps_1_3.sh /mnt/public/lcy/PTM_data_mew/PXD000612
#
# 说明：
#   - 本脚本仅做“薄封装”，实际逻辑完全委托给
#       ../script/1.run_rescore_steps_1_3.sh
#   - 该脚本内部会调用：
#       0.filter_msms_30_phospho.py
#       1.make_msms_specid.py
#       run.py
#     来完成适配磷酸化的谱图提取，并最终生成 rescore/origin_data.h5。

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workdir>" >&2
  exit 2
fi

# 当前脚本所在目录：.../mamba_DeepFLR/extract_real_spectrums
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 仓库根目录：.../mamba_DeepFLR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER_TARGET="$REPO_ROOT/script/1.run_rescore_steps_1_3.sh"

if [[ ! -f "$WRAPPER_TARGET" ]]; then
  echo "错误：找不到目标脚本: $WRAPPER_TARGET" >&2
  exit 1
fi

# 直接将参数透传给主实现脚本，避免重复维护两套逻辑
exec bash "$WRAPPER_TARGET" "$@"
