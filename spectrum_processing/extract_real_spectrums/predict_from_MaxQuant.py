#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验脚本：参考 `mamba_rescore/extract_real_spectrums` 的 H5 制作流程，
在本仓库中用“小样本 msms + mzML”快速生成可用于 MS2Int 推理的 H5。

设计目标：
1) 与参考流程一致的过滤规则：仅保留 `Modifications == Unmodified` 且 `Length <= 30`；
2) 复用本仓库已有的 Step2/3/4 实现（`step2_process_df_h5.py` 等），避免重复造轮子；
3) 发生任何异常/步骤失败时，直接退出（便于你按“遇到报错立即停止”的要求定位问题）。

输出：
- 最终 H5：`{output_dir}/{dataset_name}_batch1.h5`
  其中包含 `Sequence/annotate/Charge/Length/collision_energy/Fragmentation` 等数据集，
  可直接作为 `MS2Int/predict.py` 的 `--input`。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional


def _die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


@dataclass(frozen=True)
class WorkPaths:
    work_dir: str
    msms_filtered_dir: str
    search_dir: str
    df_h5_dir: str
    ylabel_df_dir: str
    final_dir: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_msms_tsv(msms_path: str):
    import pandas as pd

    if not os.path.isfile(msms_path):
        _die(f"找不到 msms 输入文件: {msms_path}")

    df = pd.read_csv(msms_path, sep="\t", low_memory=False)
    required_cols = [
        "Raw file",
        "Scan number",
        "Sequence",
        "Length",
        "Modifications",
        "Modified sequence",
        "Charge",
        "Mass analyzer",
        "Fragmentation",
        "Score",
        "Reverse",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _die(f"msms.txt 缺少必要列: {missing}")
    return df


def _infer_single_raw_file(df, raw_file: Optional[str]) -> str:
    raws = (
        df["Raw file"]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    if raw_file is not None:
        if raw_file not in raws:
            _die(
                f"--raw-file 指定的 Raw file 不在输入中：{raw_file}；可选值示例: {raws[:5]}"
            )
        return raw_file

    if len(raws) != 1:
        _die(
            f"输入 msms 包含多个 Raw file（数量={len(raws)}），请显式指定 --raw-file。"
        )
    return raws[0]


def _filter_unmodified_len_le_30(df):
    import pandas as pd

    # 参考 mamba_rescore 的过滤规则：
    #   Modifications == 'Unmodified' 且 Length <= 30
    lengths = pd.to_numeric(df.get("Length"), errors="coerce")
    mods = df.get("Modifications")
    keep = (mods == "Unmodified") & (lengths <= 30)
    out = df.loc[keep].copy()

    # 写回 Length 为数值型，避免后续 Step2 读入时出现混合类型
    out["Length"] = pd.to_numeric(out["Length"], errors="coerce")
    return out


def _find_mzml_path(mzml_dir: str, raw: str) -> str:
    if not os.path.isdir(mzml_dir):
        _die(f"找不到 mzML 目录: {mzml_dir}")

    # 兼容 .mzML / .mzml
    candidates = [f"{raw}.mzML", f"{raw}.mzml"]
    for name in candidates:
        p = os.path.join(mzml_dir, name)
        if os.path.isfile(p):
            return p
    _die(f"在 mzML 目录中找不到与 Raw file 对应的文件: {candidates}")
    raise AssertionError("unreachable")


def _validate_final_h5(h5_path: str) -> None:
    import h5py
    import numpy as np

    if not os.path.isfile(h5_path):
        _die(f"最终 H5 未生成或路径不存在: {h5_path}")

    required = ["Sequence", "Length", "Charge", "collision_energy", "Fragmentation"]
    with h5py.File(h5_path, "r") as f:
        missing = [k for k in required if k not in f]
        if missing:
            _die(f"最终 H5 缺少必要数据集: {missing}")

        ce = f["collision_energy"][()]
        # 允许 float/int，但不允许 NaN
        if np.issubdtype(ce.dtype, np.floating) and np.isnan(ce).any():
            _die("最终 H5 的 collision_energy 含 NaN，MS2Int 推理会因 int(NaN) 崩溃。")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="用小样本 msms+ mzML 生成可用于 MS2Int 推理的 H5（参考 mamba_rescore 流程）"
    )
    parser.add_argument(
        "--msms",
        required=True,
        help="输入 msms.txt（可为已截取的小文件，制表符分隔）",
    )
    parser.add_argument(
        "--mzml-dir",
        required=True,
        help="mzML 文件目录（需包含 <Raw file>.mzML）",
    )
    parser.add_argument(
        "--raw-file",
        default=None,
        help="可选：指定 Raw file（当 msms 中包含多个 raw 时必须指定）",
    )
    parser.add_argument(
        "--dataset-name",
        default="origin_data",
        help="输出 H5 文件名前缀（默认: origin_data）",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="工作目录（默认: <output-dir>/.work_extract_real_spectrums）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="最终输出目录（写出 {dataset}_batch1.h5）",
    )
    parser.add_argument(
        "--inner-procs",
        type=int,
        default=4,
        help="Step2 内部并行数（默认: 4；小样本建议 1~4）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Step3/4 读取/写入并行数（默认: 1；小样本建议 1）",
    )
    args = parser.parse_args(argv)

    msms_path = os.path.abspath(args.msms)
    mzml_dir = os.path.abspath(args.mzml_dir)
    final_dir = os.path.abspath(args.output_dir)
    _ensure_dir(final_dir)

    work_dir = (
        os.path.abspath(args.work_dir)
        if args.work_dir is not None
        else os.path.join(final_dir, ".work_extract_real_spectrums")
    )

    paths = WorkPaths(
        work_dir=work_dir,
        msms_filtered_dir=os.path.join(work_dir, "MSMS_filtered"),
        search_dir=os.path.join(work_dir, "Search"),
        df_h5_dir=os.path.join(work_dir, "0.process_df_h5"),
        ylabel_df_dir=os.path.join(work_dir, "1.ylabel_df"),
        final_dir=final_dir,
    )
    for p in [
        paths.msms_filtered_dir,
        paths.search_dir,
        paths.df_h5_dir,
        paths.ylabel_df_dir,
        paths.final_dir,
    ]:
        _ensure_dir(p)

    # ---- Step 0: 读取并筛选 msms（对齐参考流程） ----
    df = _read_msms_tsv(msms_path)
    raw = _infer_single_raw_file(df, args.raw_file)
    df = df.loc[df["Raw file"].astype(str) == str(raw)].copy()
    df_filtered = _filter_unmodified_len_le_30(df)

    if len(df_filtered) == 0:
        _die(
            "过滤后无可用条目（需要 Modifications==Unmodified 且 Length<=30）。"
            "你可以换一个 msms 子集，或放宽过滤规则。"
        )

    msms_filtered_path = os.path.join(paths.msms_filtered_dir, f"{raw}.txt")
    df_filtered.to_csv(msms_filtered_path, sep="\t", index=False)
    print(f"[OK] 写出过滤后的 msms: {msms_filtered_path}（行数={len(df_filtered)}）")

    # Step2 使用 meta_path 的文件名去 msms_filtered_dir 找对应 msms，因此这里建同名空文件占位
    search_placeholder = os.path.join(paths.search_dir, f"{raw}.txt")
    with open(search_placeholder, "w", encoding="utf-8"):
        pass

    mzml_path = _find_mzml_path(mzml_dir, raw)
    print(f"[OK] 匹配到 mzML: {mzml_path}")

    # ---- Step 2: mzML 读取 + 理论碎片 + 强度匹配（生成 df_h5）----
    from step2_process_df_h5 import process_pair

    ok = process_pair(
        meta_path=search_placeholder,
        mz_path=mzml_path,
        msms_root=paths.msms_filtered_dir,
        df_h5_dir=paths.df_h5_dir,
        inner_num_procs=max(1, int(args.inner_procs)),
    )
    if not ok:
        _die("Step2 处理失败（详见上方日志）。")
    print("[OK] Step2 完成")

    # ---- Step 3: 生成 train_data ----
    from step3_generate_train_data import run_step3

    config = {
        "dataset": {"name": str(args.dataset_name)},
        "paths": {
            "msms_filtered_dir": paths.msms_filtered_dir,
            "mzml_dir": mzml_dir,
            "search_dir": paths.search_dir,
            "df_h5_dir": paths.df_h5_dir,
            "ylabel_df_dir": paths.ylabel_df_dir,
            "final_dir": paths.final_dir,
        },
        "performance": {
            "num_workers": max(1, int(args.num_workers)),
            "batch_size": 400,
        },
    }
    run_step3(config)
    print("[OK] Step3 完成")

    # ---- Step 4: 合并为最终 H5 ----
    from step4_merge_final_data import run_step4

    run_step4(config)
    print("[OK] Step4 完成")

    final_h5 = os.path.join(paths.final_dir, f"{args.dataset_name}_batch1.h5")
    _validate_final_h5(final_h5)
    print(f"[OK] 最终 H5 生成并通过校验: {final_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

