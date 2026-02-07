# MS2Int 推理：缺少 causal-conv1d 导致 Mamba2 fast path 报错的修复

## 现象
- 在 `conda(mamba_dev)` 下运行 `python MS2Int/predict.py ...` 时，报错：
  - `TypeError: 'NoneType' object is not callable`
  - 堆栈位于 `mamba_ssm/ops/triton/ssd_combined.py` 内部调用 `causal_conv1d_fwd_function(...)`

## 根因
- 当前环境未安装 `causal-conv1d`，导致 Mamba2 的 mem-efficient/combined Triton 路径依赖函数为空。

## 解决方案（不额外安装包）
- 在 `MS2Int/model.py:create_block` 中为 Mamba2 自动补全配置：
  - 若用户未显式指定 `use_mem_eff_path`，则根据 `causal_conv1d_fn` 是否可用决定：
    - 缺失时 `use_mem_eff_path=False`（走非 fast path，保证可运行）
    - 存在时保持默认（可走 fast path）

## 验证
- 使用 checkpoint `.../5.tools/dia/2.pick.NCE/3.finetune_out/best_epoch_5_val_loss_0.3569_0105_062132.pth`，对 `data/MS2Int_input.h5` 推理成功，写入 `Intpredict`，shape `(103, 29, 31)`。