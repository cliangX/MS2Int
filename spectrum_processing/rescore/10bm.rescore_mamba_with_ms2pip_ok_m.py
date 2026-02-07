#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescore-Mamba（合并版 m 离子版本）: 读取 msms_specid_with_ms2pip_ok_m.tsv（原始 msms_specid + OK 特征），
并在送入 mokapot 前自动仅保留数值特征列。

与 10b.rescore_mamba_with_ms2pip_ok.py 的区别：
  - 默认输入为合并版（原始列 + OK 特征 + m 离子）
  - 读取后删除 SpecId 列，并仅保留数值列（自动过滤非数值的原始列）
  - 日志包含 tsv_variant = "merged_ok"，默认日志文件 rescore.log
  - 输出目录改为 rescore_mamba_ok_m
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from mokapot import brew
from ms2rescore.feature_generators.basic import BasicFeatureGenerator
from ms2rescore.feature_generators.maxquant import MaxQuantFeatureGenerator
from ms2rescore.rescoring_engines.mokapot import convert_psm_list
from psm_utils.io import read_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mamba 重打分（合并版：原TSV+OK特征+m离子）",
    )
    p.add_argument("-i", "--msms_path", required=True, help="输入 msms.txt 路径")
    p.add_argument(
        "-t",
        "--tsv_path",
        required=True,
        help="特征 TSV 路径（默认 msms_specid_with_ms2pip_ok_m.tsv；行序需与 PSM 一致）",
    )
    p.add_argument(
        "--drop_specid",
        action="store_true",
        default=True,
        help="若存在 SpecId 列则删除（默认启用）",
    )

    # mokapot 超参
    p.add_argument("--rng", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--folds", type=int, default=1, help="交叉验证折数（默认 1）")
    p.add_argument("--max_workers", type=int, default=1, help="并行工作数（默认 1）")

    # 阶段2特征开关
    p.add_argument(
        "--no-basic",
        dest="add_basic",
        action="store_false",
        help="阶段2不添加 Basic 特征",
    )
    p.add_argument(
        "--no-maxquant",
        dest="add_maxquant",
        action="store_false",
        help="阶段2不添加 MaxQuant 特征",
    )

    # 日志
    p.add_argument(
        "--log_path",
        default="rescore.log",
        help="JSON Lines 日志路径（默认 rescore.log，所有实验统一追加）",
    )
    p.add_argument("--overwrite", action="store_true", help="覆盖写日志（默认追加）")
    p.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")

    # 仅控制 mokapot 的日志级别
    p.add_argument(
        "--brew-log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="仅控制 mokapot 的日志级别（默认 WARNING）",
    )

    return p.parse_args()


def convert_value(colname: str, value: Any) -> Any:
    if colname.startswith("charge_"):
        return np.int8(value)
    if colname == "charge_n":
        return np.int64(value)
    if isinstance(value, (int, np.integer)):
        return np.int64(value)
    return np.float64(value)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(pd, "Timedelta") and isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(x) for x in list(obj)]
    return obj


def accepted_to_records(accepted_obj: Any) -> List[Dict[str, Any]]:
    try:
        df = pd.DataFrame(accepted_obj)
        if not df.empty:
            return [_to_builtin(r) for r in df.to_dict(orient="records")]
    except Exception:
        pass
    if isinstance(accepted_obj, list):
        return [_to_builtin(x) for x in accepted_obj]
    return [_to_builtin(accepted_obj)]


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("rescore_mamba_merged_ok_m")

    # 控制 mokapot 日志级别（独立于 -v 开关）
    def set_library_log_level(logger_name: str, level_name: str) -> None:
        import logging as _logging

        level = getattr(_logging, str(level_name).upper(), _logging.WARNING)
        lg = _logging.getLogger(logger_name)
        lg.setLevel(level)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)
        if level >= _logging.ERROR:
            lg.addHandler(_logging.NullHandler())

    level_from_env = os.getenv("MOKAPOT_LOG_LEVEL", args.brew_log_level)
    try:
        set_library_log_level("mokapot", level_from_env)
    except Exception:
        pass

    # 记录开始时间
    start_time = time.time()

    logger.info(f"读取 PSM：{args.msms_path}")
    psm_list = read_file(args.msms_path, filetype="msms")

    # 修饰标准化（与主脚本一致）
    try:
        psm_list.rename_modifications({"ox": "Oxidation"})
        psm_list.add_fixed_modifications([("U:Carbamidomethyl", ["C"])])
        psm_list.apply_fixed_modifications()
    except Exception as e:
        logger.warning(f"应用修饰失败（忽略继续）: {e}")

    logger.info(f"读取特征 TSV（合并版 m 离子）：{args.tsv_path}")
    df_raw = pd.read_csv(args.tsv_path, sep="\t")

    if args.drop_specid and "SpecId" in df_raw.columns:
        df_raw = df_raw.drop(columns=["SpecId"])  # 删除 SpecId

    # 仅保留数值列，自动过滤原始 TSV 的字符串列
    df = df_raw.select_dtypes(include=[np.number])
    dropped_cols = [c for c in df_raw.columns if c not in df.columns]
    logger.info(
        f"保留数值列 {len(df.columns)} 个，已自动过滤非数值列 {len(dropped_cols)} 个"
    )

    if len(df) != len(psm_list):
        raise ValueError(
            f"TSV 行数({len(df)}) 与 PSM 数({len(psm_list)}) 不一致。请先对齐后再运行。"
        )

    # 构造 rescoring_features（逐列类型转换）
    logger.info("构造 rescoring_features ...")
    records: List[Dict[str, Any]] = []
    cols = list(df.columns)
    for _, row in df.iterrows():
        row_dict = {col: convert_value(col, row[col]) for col in cols}
        records.append(row_dict)
    psm_list["rescoring_features"] = records

    # 如果不是 verbose 模式，抑制 mokapot 的详细日志
    if not args.verbose:
        logging.getLogger("mokapot").setLevel(logging.WARNING)

    # 阶段1：brew
    logger.info(
        f"阶段1：brew(rng={args.rng}, folds={args.folds}, max_workers={args.max_workers})"
    )
    ds1 = convert_psm_list(psm_list)
    conf1, _ = brew(ds1, rng=args.rng, folds=args.folds, max_workers=args.max_workers)

    # 阶段2：可选添加 Basic / MaxQuant
    did_add = []
    if args.add_basic:
        logger.info("阶段2：添加 Basic 特征")
        BasicFeatureGenerator().add_features(psm_list)
        did_add.append("basic")
    if args.add_maxquant:
        logger.info("阶段2：添加 MaxQuant 特征")
        MaxQuantFeatureGenerator().add_features(psm_list)
        did_add.append("maxquant")

    logger.info(
        f"阶段2：brew(rng={args.rng}, folds={args.folds}, max_workers={args.max_workers})"
    )
    ds2 = convert_psm_list(psm_list)
    conf2, _ = brew(ds2, rng=args.rng, folds=args.folds, max_workers=args.max_workers)

    # 导出 mokapot 结果 - Merged OK (m 离子版本)
    feature_tag = "_".join(did_add) if did_add else "merged_ok"
    export_dir = os.path.join(
        os.getcwd(), "rescore", "rescore_mamba_ok_m", f"mokapot_merged_ok_{feature_tag}"
    )
    os.makedirs(export_dir, exist_ok=True)
    logger.info(f"导出 mokapot 结果到: {export_dir}")
    conf2.to_txt(dest_dir=export_dir)

    accepted = conf2.accepted
    records_out = accepted_to_records(accepted)

    # 统一的日志格式
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "mamba",
        # 统一字段：features_used 明确表示所用特征类别
        "features_used": "pip,ok",
        "output_dir": os.path.abspath(export_dir),
        "m_ion": True,
        "features": {"used": did_add, "count": int(len(cols))},
        "parameters": {
            "rng": int(args.rng),
            "folds": int(args.folds),
            "max_workers": int(args.max_workers),
        },
        "results": {
            "psms": records_out[0]["psms"]
            if records_out and len(records_out) > 0
            else 0,
            "peptides": records_out[0]["peptides"]
            if records_out and len(records_out) > 0
            else 0,
            "proteins": records_out[0].get("proteins")
            if records_out and len(records_out) > 0
            else None,
        },
        "duration_seconds": round(time.time() - start_time, 2),
    }

    mode = "w" if args.overwrite else "a"
    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    with open(args.log_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(_to_builtin(entry), ensure_ascii=False) + "\n")
    logger.info(
        f"已写入日志: {args.log_path}（variant=merged_ok, accepted={len(records_out)}）"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        sys.exit(130)
