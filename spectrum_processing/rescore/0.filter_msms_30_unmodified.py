#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
过滤 msms.txt：
1) 仅保留 Modifications == 'Unmodified' 的行（去除首尾空格，大小写不敏感）
2) 仅保留 Length <= 30 的行（Length 自动转为数值，无法转换的行将丢弃）
3) 删除 Sequence 列包含 'U' 的行（'U' 代表硒，不参与标准蛋白质搜索）

用法示例：
python utils/0.filter_msms_30_unmodified.py \
-i /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_PXD000561/txt/msms.txt \
-o /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_PXD000561/output/msms.filtered_unmodified_lenle30.txt

按块大小读取（处理大文件时推荐）：
python utils/0.filter_msms_30_unmodified.py \
-i /path/to/msms.txt -o /path/to/output.txt --chunksize 200000
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import pandas as pd


def build_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        return output_path
    root, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".txt"
    return f"{root}.filtered_unmodified_lenle30{ext}"


def filter_chunk(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "in_rows": len(df),
        "kept_rows": 0,
        "removed_by_modifications": 0,
        "removed_by_length": 0,
        "removed_by_U_in_seq": 0,
    }

    required = ["Modifications", "Length", "Sequence"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列: {col}")

    # 第一步：过滤 Modifications（仅保留 Unmodified）
    mods = df["Modifications"].astype(str).str.strip().str.lower()
    is_unmod = mods.eq("unmodified")
    removed_mods = len(df) - is_unmod.sum()
    stats["removed_by_modifications"] = removed_mods

    # 第二步：过滤 Length（仅保留 Length <= 30）
    length_num = pd.to_numeric(df["Length"], errors="coerce")
    is_len_ok = length_num.le(30)
    removed_length = (~is_len_ok).sum()
    stats["removed_by_length"] = removed_length

    # 第三步：过滤 Sequence 中含 'U' 的行（删除含 U 的行）
    seq_str = df["Sequence"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
    )
    has_U = seq_str.str.contains("U", regex=False)
    is_seq_ok = ~has_U  # 只保留不含 U 的行
    removed_U = has_U.sum()
    stats["removed_by_U_in_seq"] = removed_U

    # 三个条件综合
    mask = is_unmod & is_len_ok & is_seq_ok
    kept = df.loc[mask].copy()

    stats["kept_rows"] = len(kept)
    return kept, stats


def write_out(df: pd.DataFrame, path: str, mode: str, header: bool) -> None:
    compression = "gzip" if path.endswith(".gz") else None
    df.to_csv(
        path, sep="\t", index=False, mode=mode, header=header, compression=compression
    )


def main():
    parser = argparse.ArgumentParser(
        description="过滤 msms：仅保留 Unmodified 且 Length<=30 的行"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="输入 msms.txt（制表符分隔）"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="输出文件路径（可选，不填则自动命名）"
    )
    parser.add_argument(
        "--chunksize", type=int, default=0, help="按块大小读取（大文件建议如 200000）"
    )
    parser.add_argument("--encoding", default="utf-8", help="文件编码（默认 utf-8）")
    args = parser.parse_args()

    input_path = args.input
    output_path = build_output_path(input_path, args.output)

    if not os.path.isfile(input_path):
        print(f"错误：找不到输入文件: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    total_in = 0
    total_kept = 0
    total_removed_mods = 0
    total_removed_length = 0
    total_removed_U = 0
    first_write = True

    try:
        if args.chunksize and args.chunksize > 0:
            reader = pd.read_csv(
                input_path,
                sep="\t",
                encoding=args.encoding,
                chunksize=args.chunksize,
                low_memory=False,
            )
            for chunk in reader:
                kept, st = filter_chunk(chunk)
                total_in += st["in_rows"]
                total_kept += st["kept_rows"]
                total_removed_mods += st["removed_by_modifications"]
                total_removed_length += st["removed_by_length"]
                total_removed_U += st["removed_by_U_in_seq"]
                write_out(
                    kept,
                    output_path,
                    mode=("w" if first_write else "a"),
                    header=first_write,
                )
                first_write = False
        else:
            df = pd.read_csv(
                input_path, sep="\t", encoding=args.encoding, low_memory=False
            )
            kept, st = filter_chunk(df)
            total_in += st["in_rows"]
            total_kept += st["kept_rows"]
            total_removed_mods += st["removed_by_modifications"]
            total_removed_length += st["removed_by_length"]
            total_removed_U += st["removed_by_U_in_seq"]
            write_out(kept, output_path, mode="w", header=True)

        pct = (total_kept / total_in * 100.0) if total_in else 0.0
        print(f"\n{'=' * 60}")
        print("过滤完成！统计信息：")
        print(f"{'=' * 60}")
        print(f"输入总行数：                {total_in:,}")
        print(f"  - 删除（Modifications）： {total_removed_mods:,}")
        print(f"  - 删除（Length > 30）：   {total_removed_length:,}")
        print(f"  - 删除（Sequence含'U'）： {total_removed_U:,}")
        print(f"保留行数：                  {total_kept:,} ({pct:.2f}%)")
        print(f"{'=' * 60}")
        print(f"输出文件：{output_path}")
        print(f"{'=' * 60}\n")
    except UnicodeDecodeError:
        print(
            "编码错误：请尝试指定 --encoding 如 'latin-1' 或 'gb18030'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
