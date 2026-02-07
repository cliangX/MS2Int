# 从 mzML 提取 Fragmentation（HCD/CID）并写入最终 H5

## 背景
- 之前 H5 中的 `Fragmentation` 字段来自 `msms.txt` 的 `Fragmentation` 列（Step2 merge 后由 Step4 写入）。
- 为避免 `mzML` 中 activation method 的长字符串（如 `beam-type collision-induced dissociation`）难以直接映射，改为使用 PyOpenMS 的 short string。

## 实现位置
- 文件：`spectrum_processing/extract_real_spectrums/step2_process_df_h5.py`
- 在读取 mzML 的循环中，对每个 MS2 spectrum 的 precursor 追加：
  - `short = [x.decode() for x in precursor.getActivationMethodsAsShortString()]`
  - `frag = short[0] if short else ""`
  - 写入 `mz_df["Fragmentation_mzml"]`，merge 后优先用该列覆盖 `combined_df["Fragmentation"]`。

## 行为与验证
- 若 mzML 能提取到短字符串（例如 `HCD`/`CID`），则覆盖 msms 中的 `Fragmentation`。
- 小样本验证：对 `HF2_MS_14359_BH-E_1_22072021.mzML` 提取时日志显示 `Fragmentation检查: 使用mzML覆盖=103/103`，最终 H5 中 `Fragmentation` 唯一值为 `b'HCD'`。

## 备注
- 该做法对 HCD/CID 等常见碎裂方式最直接；若遇到多 activation method，当前取 `short[0]`。