#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 HDF5 文件生成并写入 SpecId 数据集的工具。

SpecId 规则:
  SpecId = "{Raw_file}-{MS2_Scan_Number}-{Sequence}-{Charge}"

处理细节:
  - 从 H5 读取数据集: Raw_file, MS2_Scan_Number, annotate(序列), Charge
  - 将字节型转为 UTF-8 字符串
  - 将 Sequence 中的 "[Carbamidomethyl]" 替换为 "[UNIMOD:4]"
  - 若已存在 SpecId，默认覆盖重建（可用 --no-overwrite 跳过）

用法示例:
  python ok_feature/add_SpecId2h5.py \
      --h5_path /path/to/PXDxxxxxx.h5 \
      --verbose

可选参数:
  --raw_key/--scan_key/--seq_key/--charge_key 可自定义各数据集键名（默认与脚本一致）
  --no-replace-carba 关闭 [Carbamidomethyl] -> [UNIMOD:4] 替换
  --no-overwrite     若 H5 已存在 SpecId，则跳过而非覆盖
  --dry_run          仅检查与预览，不写回
"""

import argparse
import logging
import os
import sys
from typing import List

import h5py
import numpy as np


def _decode_str_array(arr) -> List[str]:
    out: List[str] = []
    for v in arr:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def _to_int_str_array(arr) -> List[str]:
    # 将数值数组转换为整数字符串（scan、charge 常为整数）
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.integer):
            return [str(int(x)) for x in arr]
        if np.issubdtype(arr.dtype, np.floating):
            return [str(int(round(float(x)))) for x in arr]
    # 回退: 逐个转 int 再转 str
    out: List[str] = []
    for v in arr:
        try:
            out.append(str(int(round(float(v)))))
        except Exception:
            out.append(str(v))
    return out


def build_specid(
    raw_files: List[str],
    scans: List[str],
    seqs: List[str],
    charges: List[str],
    replace_carba: bool = True,
) -> List[str]:
    if replace_carba:
        seqs = [s.replace("[Carbamidomethyl]", "[UNIMOD:4]") for s in seqs]
    return [
        f"{rf}-{msn}-{seq}-{ch}"
        for rf, msn, seq, ch in zip(raw_files, scans, seqs, charges)
    ]


def add_specid_to_h5(
    h5_path: str,
    raw_key: str = "Raw_file",
    scan_key: str = "MS2_Scan_Number",
    seq_key: str = "annotate",
    charge_key: str = "Charge",
    replace_carba: bool = True,
    overwrite: bool = True,
    dry_run: bool = False,
) -> None:
    logger = logging.getLogger("add_SpecId2h5")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 文件不存在: {h5_path}")

    with h5py.File(h5_path, "a") as h5f:
        keys = list(h5f.keys())
        logger.info(f"H5 文件键: {keys}")

        for k in [raw_key, scan_key, seq_key, charge_key]:
            if k not in h5f:
                raise KeyError(f"H5 缺少数据集: {k}")

        raw_arr = h5f[raw_key][:]
        seq_arr = h5f[seq_key][:]
        scan_arr = h5f[scan_key][:]
        charge_arr = h5f[charge_key][:]

        raw_list = _decode_str_array(raw_arr)
        seq_list = _decode_str_array(seq_arr)
        scan_list = _to_int_str_array(scan_arr)
        charge_list = _to_int_str_array(charge_arr)

        n = len(raw_list)
        if not (len(seq_list) == len(scan_list) == len(charge_list) == n):
            raise ValueError(
                f"长度不一致: Raw={len(raw_list)}, Scan={len(scan_list)}, Seq={len(seq_list)}, Charge={len(charge_list)}"
            )

        logger.info(f"将构建 SpecId 条目数: {n}")
        specids = build_specid(
            raw_list, scan_list, seq_list, charge_list, replace_carba=replace_carba
        )

        # 预览前 5 条
        for i in range(min(5, n)):
            logger.debug(f"示例[{i}]: {specids[i]}")

        if dry_run:
            logger.info("dry-run 模式：不写回 H5，仅打印检查信息。")
            return

        if "SpecId" in h5f:
            if overwrite:
                logger.info("H5 已存在 'SpecId'，将覆盖重建。")
                del h5f["SpecId"]
            else:
                logger.info("H5 已存在 'SpecId'，根据 --no-overwrite 跳过写入。")
                return

        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset(
            "SpecId", data=np.array(specids, dtype=object), dtype=str_dtype
        )
        logger.info(f"完成：已写入 SpecId（{n} 条） -> {h5_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="为 HDF5 添加 SpecId 数据集的工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s -i /path/to/PXDxxxxxx.h5 -v
  %(prog)s --h5_path data.h5 --no-overwrite --no-replace-carba
  %(prog)s -i data.h5 --raw_key Raw_file --scan_key MS2_Scan_Number --seq_key annotate --charge_key Charge
""",
    )

    p.add_argument("-i", "--h5_path", required=True, help="输入 H5 文件路径")
    p.add_argument("--raw_key", default="Raw_file", help="Raw 文件名数据集键名")
    p.add_argument("--scan_key", default="MS2_Scan_Number", help="MS2 扫描号数据集键名")
    p.add_argument("--seq_key", default="annotate", help="序列数据集键名")
    p.add_argument("--charge_key", default="Charge", help="电荷数据集键名")

    p.add_argument(
        "--no-replace-carba",
        dest="replace_carba",
        action="store_false",
        help="不进行 [Carbamidomethyl] -> [UNIMOD:4] 的替换",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="若已存在 SpecId 则跳过写入（默认覆盖重建）",
    )
    p.add_argument("--dry_run", action="store_true", help="试运行，仅检查不写回")
    p.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")

    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("add_SpecId2h5")

    try:
        add_specid_to_h5(
            h5_path=args.h5_path,
            raw_key=args.raw_key,
            scan_key=args.scan_key,
            seq_key=args.seq_key,
            charge_key=args.charge_key,
            replace_carba=args.replace_carba,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"失败: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
