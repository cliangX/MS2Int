#!/usr/bin/env bash
# PTM 磷酸化微调（MS2Int train40 / fine_tune_novat）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export HDF5_USE_FILE_LOCKING=FALSE

DATA_H5="${REPO_ROOT}/data/ptm_finetune/PTM_train_data_train40.h5"
OUT_DIR="${REPO_ROOT}/ptm_finetune/checkpoints"
LOG_PATH="${REPO_ROOT}/ptm_finetune/logs/finetune.log"

# 续训：上一轮最佳 epoch7 微调权重（val_loss=0.3156）
RESUME_CKPT="${REPO_ROOT}/ptm_finetune/checkpoints/model_epoch_7_val_loss_0.3156_0626_221157.pth"

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/ptm_finetune/logs"

cd "${REPO_ROOT}/MS2Int"

CONDA_ENV="${CONDA_ENV:-mamba_dev}"

conda run --no-capture-output -n "${CONDA_ENV}" python fine_tune_novat.py \
  --experiment_name "MS2Int_PTM_Phospho_FT" \
  --world_size 1 \
  --resume "${RESUME_CKPT}" \
  --train_data_path "${DATA_H5}" \
  --checkpoint_path "${OUT_DIR}" \
  --log_path "${LOG_PATH}" \
  --train_batch_size 512 \
  --val_batch_size 1024 \
  --num_workers 8 \
  --train_data_size 0.99 \
  --learning_rate 3e-5 \
  --warmup_iters 500 \
  --max_epochs 100 \
  --freeze_last_k 1 \
  --port 29606
