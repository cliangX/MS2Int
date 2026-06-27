#!/usr/bin/env python3
"""Filter msms.txt: keep Unmodified, Length<=40, no selenocysteine (U)."""

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
    return f"{root}.filtered_unmodified_lenle40{ext}"


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
            raise ValueError(f"Missing required column: {col}")

    mods = df["Modifications"].astype(str).str.strip().str.lower()
    is_unmod = mods.eq("unmodified")
    removed_mods = len(df) - is_unmod.sum()
    stats["removed_by_modifications"] = removed_mods

    length_num = pd.to_numeric(df["Length"], errors="coerce")
    is_len_ok = length_num.le(40)
    removed_length = (~is_len_ok).sum()
    stats["removed_by_length"] = removed_length

    seq_str = df["Sequence"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
    )
    has_U = seq_str.str.contains("U", regex=False)
    is_seq_ok = ~has_U
    removed_U = has_U.sum()
    stats["removed_by_U_in_seq"] = removed_U

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--chunksize", type=int, default=0)
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    input_path = args.input
    output_path = build_output_path(input_path, args.output)

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
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
    print(f"Kept {total_kept:,}/{total_in:,} ({pct:.1f}%) -> {output_path}")


if __name__ == "__main__":
    main()
