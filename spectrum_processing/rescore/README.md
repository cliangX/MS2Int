#!/usr/bin/env markdown
# Rescore（Mamba + 全特征，含 m 离子）

本目录从 `/mnt/data_nas/lcy/project_MS2predict/5.tools/mamba_rescore` 抽取了 **“Mamba + 全特征”** 的重打分实现，目标是把 **Mamba 预测（Intpredict）** 与 **MS2PIP 特征 + OK 特征 + m 离子扩展（203维）** 组合起来，最终使用 **mokapot** 进行重打分。

这里的“全特征”具体指：
- **MS2PIP 风格特征（含 m 离子）**：`6m.calculator_ms2pip_feature_m.py`
- **OK 特征（203维，含 m 离子）**：`7m.calculator_ok_feature_m.py` + `ok_cpm.py`
- **Basic / MaxQuant 二阶段特征**：在 `10bm.rescore_mamba_with_ms2pip_ok_m.py` 中通过 `ms2rescore` 自动添加（默认开启）

---

## 📦 已抽取文件

- `spectrum_processing/rescore/0.filter_msms_30_unmodified.py`：过滤 MaxQuant `msms.txt`（Unmodified + Length<=30 + 去掉含 U 的序列）
- `spectrum_processing/rescore/1.make_msms_specid.py`：生成 `msms_specid.tsv`（按 Raw file/Scan number/Sequence/Charge 拼 `SpecId`）
- `spectrum_processing/rescore/5.add_SpecId_2_h5.py`：为 H5 写入 `SpecId` 数据集（供后续按 SpecId 对齐）
- `spectrum_processing/rescore/6m.calculator_ms2pip_feature_m.py`：计算 MS2PIP 风格特征（**含 m 离子**）
- `spectrum_processing/rescore/7m.calculator_ok_feature_m.py`：计算 OK 特征（**203维含 m 离子**），并输出“合并版 TSV”
- `spectrum_processing/rescore/10bm.rescore_mamba_with_ms2pip_ok_m.py`：使用 mokapot 进行重打分（读取“合并版 TSV”）
- `spectrum_processing/rescore/ok_cpm.py`：OK 特征计算核心实现

---

## ✅ 输入/输出约定

### 输入（工作目录 WORKDIR）

WORKDIR 需要包含：
```
WORKDIR/
├── txt/
│   └── msms.txt
└── mzml/
    ├── *.mzML
    └── ...
```

### 中间/输出（写入 WORKDIR/rescore/）

主要产物：
- `rescore/1.msms_filtered_unmodified_lenle30.txt`
- `rescore/msms_specid.tsv`
- `rescore/rescore_batch1.h5`（由 `spectrum_processing/unmodficaiton/run.py` 生成）
- `rescore/rescore.h5`（由 `MS2Int/predict.py` 写入 `Intpredict`）
- `rescore/msms_specid_with_MS2PIP_m.tsv`
- `rescore/msms_specid_with_ms2pip_ok_m.tsv`（**最终用于 mokapot 的“全特征合并版”**）

---

## 🚀 推荐运行流程（手动分步）

下面示例默认你在 MS2Int 仓库根目录运行（`/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int`），并且使用 conda 环境 `mamba_dev`。

1) 过滤 `msms.txt`
```bash
conda run -n "mamba_dev" python "spectrum_processing/rescore/0.filter_msms_30_unmodified.py" \
  -i "/path/to/WORKDIR/txt/msms.txt" \
  -o "/path/to/WORKDIR/rescore/1.msms_filtered_unmodified_lenle30.txt"
```

2) 生成 `msms_specid.tsv`
```bash
conda run -n "mamba_dev" python "spectrum_processing/rescore/1.make_msms_specid.py" \
  "/path/to/WORKDIR/rescore/1.msms_filtered_unmodified_lenle30.txt" \
  "/path/to/WORKDIR/rescore/msms_specid.tsv"
```

3) 制作 H5（真实谱图 + train_data）
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/spectrum_processing/unmodficaiton/run.py" \
  --msms "rescore/1.msms_filtered_unmodified_lenle30.txt" \
  --mzml-dir "mzml" \
  --dataset-name "rescore" \
  --output-dir "rescore" \
  --final-h5 "rescore/rescore_batch1.h5"
```

4) 运行 MS2Int 推理（写入 `Intpredict`）
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/MS2Int/predict.py" \
  --ckpt "/path/to/model.pth" \
  --input "rescore/rescore_batch1.h5" \
  --output "rescore/rescore.h5"
```

5) 为 `rescore.h5` 写入 `SpecId`
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/spectrum_processing/rescore/5.add_SpecId_2_h5.py" \
  --h5_path "rescore/rescore.h5"
```

6) 计算 MS2PIP 特征（含 m 离子）
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/spectrum_processing/rescore/6m.calculator_ms2pip_feature_m.py" \
  --h5_path "rescore/rescore.h5" \
  --tsv_path "rescore/msms_specid.tsv" \
  --output "rescore/msms_specid_with_MS2PIP_m.tsv"
```

7) 计算 OK 特征（203维含 m 离子）并生成“合并版 TSV”
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/spectrum_processing/rescore/7m.calculator_ok_feature_m.py" \
  --h5_path "rescore/rescore.h5" \
  --tsv_path "rescore/msms_specid_with_MS2PIP_m.tsv" \
  --output "rescore/msms_specid_with_ms2pip_ok_m.tsv"
```

8) mokapot 重打分（Mamba + 全特征）
```bash
cd "/path/to/WORKDIR"
conda run -n "mamba_dev" python "/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/spectrum_processing/rescore/10bm.rescore_mamba_with_ms2pip_ok_m.py" \
  --msms_path "rescore/1.msms_filtered_unmodified_lenle30.txt" \
  --tsv_path "rescore/msms_specid_with_ms2pip_ok_m.tsv" \
  --rng 42 --folds 2 --max_workers 2 \
  --log_path "rescore/logs/rescore.log" \
  -v
```

输出目录默认在：
`WORKDIR/rescore/rescore_mamba_ok_m/`

---

## 🧩 依赖说明

除 MS2Int 本身依赖外，本流程额外需要：
- `mokapot`
- `ms2rescore`
- `psm-utils`

如果环境缺包，建议在 `mamba_dev` 中安装（示例）：
```bash
conda run -n "mamba_dev" python -m pip install --no-cache-dir mokapot ms2rescore psm-utils
```

