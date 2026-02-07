# 从 MaxQuant `msms.txt` 生成 MS2Int 推理输入 H5

## 推理入口与输入格式
- 推理脚本：`MS2Int/predict.py`（读取输入 H5，写入/覆盖 `Intpredict` 数据集）。
- 输入 H5 在 `MS2Int/preprocess.py:data_read` 中定义，推理最少需要以下数据集：
  - `Sequence`（或 `annotate`）：肽段序列，修饰用 `[]` 表示（如 `M[Oxidation]`、`S[Phospho]`、`C[Carbamidomethyl]`、`[Acetyl]-`）
  - `Length`：肽段长度
  - `Charge`：电荷
  - `collision_energy`：碰撞能量（建议取离散集合：`10,20,23,25,26,27,28,29,30,35,40,42`）
  - `Fragmentation`：碎裂方式（`HCD`/`CID`）

## 推荐转换链路（复用现有脚本）
利用 `data_processing/step4_convert_to_mamba_h5.py`（CSV → H5），只需要先把 `msms.txt` 整理成最小 CSV：
- 必需列：`SourceFile, Fspectrum, PP.Charge, key_x, PEP.StrippedSequence`
- 列映射（来自 `msms.txt`）：
  - `SourceFile` ← `Raw file`
  - `Fspectrum` ← `Scan number`
  - `PP.Charge` ← `Charge`
  - `PEP.StrippedSequence` ← `Sequence`
  - `key_x` ← 由 `Modified sequence` 清洗得到的 DeepFLR 编码串（1/2/3/4 分别表示 Phospho/Oxidation/Carbamidomethyl/Acetyl）

示例（生成 CSV）：
```bash
python - <<'PY'
import pandas as pd
msms = "txt/msms.txt"  # 你的 MaxQuant msms.txt
out_csv = "stepX_msms_minimal.csv"

df = pd.read_table(msms, sep='\t', low_memory=False)

out = pd.DataFrame({
    "SourceFile": df["Raw file"].astype(str),
    "Fspectrum": df["Scan number"].astype(str),
    "PP.Charge": pd.to_numeric(df["Charge"], errors="coerce").fillna(-1).astype(int),
    "PEP.StrippedSequence": df["Sequence"].astype(str),
})

key_x = df["Modified sequence"].astype(str)
key_x = key_x.str.replace("_", "", regex=False)
repls = [
    ("(Phospho (STY))", "1"),
    ("(Phospho(Y))", "1"),
    ("(Phospho(S))", "1"),
    ("(Phospho(T))", "1"),
    ("(Phospho (Y))", "1"),
    ("(Phospho (S))", "1"),
    ("(Phospho (T))", "1"),
    ("(Oxidation (M))", "2"),
    ("(Acetyl (Protein N-term))", "4"),
]
for a, b in repls:
    key_x = key_x.str.replace(a, b, regex=False)
key_x = key_x.str.replace("C", "C3", regex=False)

out["key_x"] = key_x
out = out.drop_duplicates(subset=["SourceFile", "Fspectrum", "PP.Charge", "key_x"]).reset_index(drop=True)

out.to_csv(out_csv, index=False)
print("写出:", out_csv, "行数:", len(out))
PY
```

CSV → H5：
```bash
python "data_processing/step4_convert_to_mamba_h5.py" \
  --input "stepX_msms_minimal.csv" \
  --output "mamba_input.h5" \
  --collision_energy 27 \
  --fragmentation HCD \
  --quiet
```

H5 → 推理：
```bash
python "MS2Int/predict.py" \
  --ckpt "/path/to/model.pth" \
  --input "mamba_input.h5" \
  --output "mamba_input.h5"
```

## 关键注意事项
- `Length>30` 的样本在 `predict.py` 中会被跳过推理并在输出中用 0 填充。
- `Charge` 建议限制在 `1..6`，超出会被编码为 0（等价于“未知”）。
- `collision_energy` 建议限制在离散集合，否则同样会被编码为 0。