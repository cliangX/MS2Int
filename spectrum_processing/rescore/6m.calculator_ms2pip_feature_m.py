#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MS2PIP feature calculator with m-ion support (quiet mode)

This script explicitly suppresses TensorFlow/absl GPU init warnings that may
appear in some environments when spawned in multiple worker processes. We do
not use TensorFlow here, so disable GPU and lower TF log level early.
"""

# Suppress TensorFlow/absl GPU init warnings before any possible import
import os as _os  # noqa: E402

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=INFO,2=WARNING,3=ERROR
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # disable GPU visibility for TF
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoid extra oneDNN logs
"""
使用方式:
  # 指定输出文件和线程数
  cd /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_PXD041529/raw/output/
  python ok_feature/ms2pip_feature_calculator.py \
      --h5_path PXD041529.h5 \
      --tsv_path msms_specid.tsv \
      --output  msms_specid_with_features.tsv \
      --workers 8 -v

"""

import argparse
import logging as _logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# 抑制 Numba 的 DEBUG 日志
_logging.getLogger("numba").setLevel(_logging.WARNING)

# 可选: 使用 numba 加速部分运算（若不可用则回退到纯 numpy 实现）
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _mse_numba(x, y):
        return np.mean((x - y) ** 2)

    @njit(cache=True, fastmath=True)
    def _cos_numba(x, y):
        nx = np.sqrt((x * x).sum())
        ny = np.sqrt((y * y).sum())
        return 0.0 if nx == 0.0 or ny == 0.0 else (x * y).sum() / (nx * ny)
else:

    def _mse_numba(x, y):
        return float(np.mean((x - y) ** 2))

    def _cos_numba(x, y):
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        return 0.0 if nx == 0.0 or ny == 0.0 else float(np.dot(x, y) / (nx * ny))


def calculate_ms2pip_features(intpredict, train_data, length):
    """
    计算完整的 MS2PIP 特征（包含 m 离子）

    Args:
        intpredict: 预测强度矩阵
        train_data: 观测强度矩阵
        length: 有效长度

    Returns:
        dict: 特征字典
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # b/y 离子强度提取
        targ_b = np.concatenate([intpredict[:length, 0], intpredict[:length, 1]])
        targ_y = np.concatenate([intpredict[:length, 2], intpredict[:length, 3]])
        # m 离子强度提取（通道 5, 6, 7）
        targ_m = np.concatenate(
            [intpredict[:length, 5], intpredict[:length, 6], intpredict[:length, 7]]
        )
        targ_all = np.concatenate([targ_b, targ_y])

        pred_b = np.concatenate([train_data[:length, 0], train_data[:length, 1]])
        pred_y = np.concatenate([train_data[:length, 2], train_data[:length, 3]])
        # m 离子强度提取（通道 5, 6, 7）
        pred_m = np.concatenate(
            [train_data[:length, 5], train_data[:length, 6], train_data[:length, 7]]
        )
        pred_all = np.concatenate([pred_b, pred_y])

        # 统一归一化：除以 pred_all 最大值
        max_pred = float(np.max(pred_all))
        if max_pred > 0.0:  # 避免除零错误
            pred_b = pred_b / max_pred
            pred_y = pred_y / max_pred
            pred_m = pred_m / max_pred
            pred_all = pred_all / max_pred

        def _spearman(x, y):
            if len(x) < 2:
                return 0.0
            try:
                return float(
                    np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
                )
            except:
                return 0.0

        eps = 0.001
        log = lambda x: np.log2(x + eps).clip(np.log2(eps))

        # log 空间计算
        tb_l, ty_l, ta_l = log(targ_b), log(targ_y), log(targ_all)
        pb_l, py_l, pa_l = log(pred_b), log(pred_y), log(pred_all)
        # m 离子 log 空间
        tm_l = log(targ_m)
        pm_l = log(pred_m)

        # 差异计算
        adb, ady, ada = (
            np.abs(targ_b - pred_b),
            np.abs(targ_y - pred_y),
            np.abs(targ_all - pred_all),
        )
        adb_l, ady_l, ada_l = (
            np.abs(tb_l - pb_l),
            np.abs(ty_l - py_l),
            np.abs(ta_l - pa_l),
        )
        # m 离子差异计算
        adm = np.abs(targ_m - pred_m)
        adm_l = np.abs(tm_l - pm_l)

        features = {
            # log 空间特征
            "spec_pearson_norm": np.corrcoef(ta_l, pa_l)[0, 1],
            "ionb_pearson_norm": np.corrcoef(tb_l, pb_l)[0, 1],
            "iony_pearson_norm": np.corrcoef(ty_l, py_l)[0, 1],
            "spec_mse_norm": _mse_numba(ta_l, pa_l),
            "ionb_mse_norm": _mse_numba(tb_l, pb_l),
            "iony_mse_norm": _mse_numba(ty_l, py_l),
            "min_abs_diff_norm": np.min(ada_l),
            "max_abs_diff_norm": np.max(ada_l),
            "abs_diff_Q1_norm": np.quantile(ada_l, 0.25),
            "abs_diff_Q2_norm": np.quantile(ada_l, 0.5),
            "abs_diff_Q3_norm": np.quantile(ada_l, 0.75),
            "mean_abs_diff_norm": np.mean(ada_l),
            "std_abs_diff_norm": np.std(ada_l),
            "ionb_min_abs_diff_norm": np.min(adb_l),
            "ionb_max_abs_diff_norm": np.max(adb_l),
            "ionb_abs_diff_Q1_norm": np.quantile(adb_l, 0.25),
            "ionb_abs_diff_Q2_norm": np.quantile(adb_l, 0.5),
            "ionb_abs_diff_Q3_norm": np.quantile(adb_l, 0.75),
            "ionb_mean_abs_diff_norm": np.mean(adb_l),
            "ionb_std_abs_diff_norm": np.std(adb_l),
            "iony_min_abs_diff_norm": np.min(ady_l),
            "iony_max_abs_diff_norm": np.max(ady_l),
            "iony_abs_diff_Q1_norm": np.quantile(ady_l, 0.25),
            "iony_abs_diff_Q2_norm": np.quantile(ady_l, 0.5),
            "iony_abs_diff_Q3_norm": np.quantile(ady_l, 0.75),
            "iony_mean_abs_diff_norm": np.mean(ady_l),
            "iony_std_abs_diff_norm": np.std(ady_l),
            "dotprod_norm": np.dot(ta_l, pa_l),
            "dotprod_ionb_norm": np.dot(tb_l, pb_l),
            "dotprod_iony_norm": np.dot(ty_l, py_l),
            "cos_norm": _cos_numba(ta_l, pa_l),
            "cos_ionb_norm": _cos_numba(tb_l, pb_l),
            "cos_iony_norm": _cos_numba(ty_l, py_l),
            # m 离子 log 空间特征（11 个）
            "ionm_pearson_norm": np.corrcoef(tm_l, pm_l)[0, 1],
            "ionm_mse_norm": _mse_numba(tm_l, pm_l),
            "ionm_min_abs_diff_norm": np.min(adm_l),
            "ionm_max_abs_diff_norm": np.max(adm_l),
            "ionm_abs_diff_Q1_norm": np.quantile(adm_l, 0.25),
            "ionm_abs_diff_Q2_norm": np.quantile(adm_l, 0.5),
            "ionm_abs_diff_Q3_norm": np.quantile(adm_l, 0.75),
            "ionm_mean_abs_diff_norm": np.mean(adm_l),
            "ionm_std_abs_diff_norm": np.std(adm_l),
            "ionm_dotprod_norm": np.dot(tm_l, pm_l),
            "ionm_cos_norm": _cos_numba(tm_l, pm_l),
            # 线性空间特征
            "spec_pearson": np.corrcoef(targ_all, pred_all)[0, 1],
            "ionb_pearson": np.corrcoef(targ_b, pred_b)[0, 1],
            "iony_pearson": np.corrcoef(targ_y, pred_y)[0, 1],
            "spec_spearman": _spearman(targ_all, pred_all),
            "ionb_spearman": _spearman(targ_b, pred_b),
            "iony_spearman": _spearman(targ_y, pred_y),
            "spec_mse": _mse_numba(targ_all, pred_all),
            "ionb_mse": _mse_numba(targ_b, pred_b),
            "iony_mse": _mse_numba(targ_y, pred_y),
            "min_abs_diff_iontype": 0 if np.min(adb) <= np.min(ady) else 1,
            "max_abs_diff_iontype": 0 if np.max(adb) >= np.max(ady) else 1,
            "min_abs_diff": np.min(ada),
            "max_abs_diff": np.max(ada),
            "abs_diff_Q1": np.quantile(ada, 0.25),
            "abs_diff_Q2": np.quantile(ada, 0.5),
            "abs_diff_Q3": np.quantile(ada, 0.75),
            "mean_abs_diff": np.mean(ada),
            "std_abs_diff": np.std(ada),
            "ionb_min_abs_diff": np.min(adb),
            "ionb_max_abs_diff": np.max(adb),
            "ionb_abs_diff_Q1": np.quantile(adb, 0.25),
            "ionb_abs_diff_Q2": np.quantile(adb, 0.5),
            "ionb_abs_diff_Q3": np.quantile(adb, 0.75),
            "ionb_mean_abs_diff": np.mean(adb),
            "ionb_std_abs_diff": np.std(adb),
            "iony_min_abs_diff": np.min(ady),
            "iony_max_abs_diff": np.max(ady),
            "iony_abs_diff_Q1": np.quantile(ady, 0.25),
            "iony_abs_diff_Q2": np.quantile(ady, 0.5),
            "iony_abs_diff_Q3": np.quantile(ady, 0.75),
            "iony_mean_abs_diff": np.mean(ady),
            "iony_std_abs_diff": np.std(ady),
            "dotprod": np.dot(targ_all, pred_all),
            "dotprod_ionb": np.dot(targ_b, pred_b),
            "dotprod_iony": np.dot(targ_y, pred_y),
            "cos": _cos_numba(targ_all, pred_all),
            "cos_ionb": _cos_numba(targ_b, pred_b),
            "cos_iony": _cos_numba(targ_y, pred_y),
            # m 离子线性空间特征（12 个）
            "ionm_pearson": np.corrcoef(targ_m, pred_m)[0, 1],
            "ionm_spearman": _spearman(targ_m, pred_m),
            "ionm_mse": _mse_numba(targ_m, pred_m),
            "ionm_min_abs_diff": np.min(adm),
            "ionm_max_abs_diff": np.max(adm),
            "ionm_abs_diff_Q1": np.quantile(adm, 0.25),
            "ionm_abs_diff_Q2": np.quantile(adm, 0.5),
            "ionm_abs_diff_Q3": np.quantile(adm, 0.75),
            "ionm_mean_abs_diff": np.mean(adm),
            "ionm_std_abs_diff": np.std(adm),
            "ionm_dotprod": np.dot(targ_m, pred_m),
            "ionm_cos": _cos_numba(targ_m, pred_m),
        }

        return {k: (0.0 if np.isnan(v) else float(v)) for k, v in features.items()}


def process_spec_batch(spec_batch, sid2idx, h5_path, predictor_name="Intpredict"):
    """
    批量处理光谱数据（子进程内按需读取 H5 数据）

    Args:
        spec_batch: 光谱ID批次
        sid2idx: 光谱ID到索引的映射
        h5_path: HDF5 文件路径
        predictor_name: 预测数据集名称（默认 Intpredict）

    Returns:
        dict: 批次处理结果，键为 SpecId，值为特征字典
    """
    results = {}
    # 每个子进程各自打开 HDF5，避免主进程大矩阵拷贝
    with h5py.File(h5_path, "r") as h5f:
        intpred_ds = h5f[predictor_name]
        train_ds = h5f["train_data"]
        lengths_ds = h5f["Length"]
        for sid in spec_batch:
            if sid in sid2idx:
                idx = sid2idx[sid]
                try:
                    intpred = intpred_ds[idx]
                    train = train_ds[idx]
                    length = int(lengths_ds[idx])
                    feats = calculate_ms2pip_features(intpred, train, length)
                    results[sid] = feats
                except Exception as e:
                    print(f"处理 {sid} 时发生错误: {e}")
                    results[sid] = {}
            else:
                results[sid] = {}
    return results


def chunk_list(lst, chunk_size):
    """
    将列表分成指定大小的批次

    Args:
        lst: 输入列表
        chunk_size: 批次大小

    Yields:
        批次列表
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def process_spectra_multithreaded(
    spec_ids_list,
    sid2idx,
    h5_path,
    max_workers=8,
    batch_size=None,
    predictor_name="Intpredict",
):
    """
    使用多进程处理光谱数据（每个子进程独立打开 H5）

    Args:
        spec_ids_list: 光谱ID列表
        sid2idx: 光谱ID到索引的映射
        max_workers: 最大工作线程数

    Returns:
        dict: 所有光谱的特征字典
    """
    # 计算合适的批次大小
    total_specs = len(spec_ids_list)
    if batch_size is None:
        batch_size = max(1, total_specs // (max_workers * 4))
        batch_size = min(batch_size, 2000)  # 限制最大批次大小避免内存问题

    # 执行信息不打印，保持控制台整洁

    # 分批
    spec_batches = list(chunk_list(spec_ids_list, batch_size))

    features_all = {}

    # 创建进度条
    pbar = tqdm(total=total_specs, desc="Processing spectra", unit="spec")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_spec_batch, batch, sid2idx, h5_path, predictor_name
            ): batch
            for batch in spec_batches
        }

        # 收集结果
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                features_all.update(batch_results)
                pbar.update(len(batch))
            except Exception as exc:
                print(f"批次处理失败: {exc}")
                # 对失败的批次中的每个spec_id添加空特征
                for sid in batch:
                    features_all[sid] = {}
                pbar.update(len(batch))

    pbar.close()
    return features_all


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="MS2PIP 特征计算工具（含 m 离子）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --h5_path /path/to/data.h5 --tsv_path /path/to/msms.tsv
  %(prog)s -i /path/to/data.h5 -t /path/to/msms.tsv --workers 16
  %(prog)s --h5_path data.h5 --tsv_path msms.tsv --output result.tsv --workers 32

注意事项:
  - H5文件必须包含: SpecId, Intpredict, train_data, Length 数据集
  - TSV文件必须包含: SpecId 列
  - 默认会覆盖原TSV文件，使用 --output 指定新的输出文件
        """,
    )

    parser.add_argument(
        "--h5_path", "-i", type=str, required=True, help="HDF5数据文件路径 (必需)"
    )

    parser.add_argument(
        "--tsv_path",
        "-t",
        type=str,
        required=True,
        help="TSV输入文件路径，必须包含SpecId列 (必需)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出文件路径 (可选，默认覆盖输入TSV文件)",
    )

    parser.add_argument(
        "--predictor_name",
        type=str,
        default="Intpredict",
        help="h5文件中预测结果的数据集名称（默认 Intpredict，可指定 tool1/tool2 等）",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="mamba",
        help="模型名称，用于自动生成输出文件名（默认 mamba）",
    )

    parser.add_argument(
        "--workers", "-w", type=int, default=16, help="并行处理的工作线程数 (默认: 8)"
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=None,
        help="批处理大小 (可选，默认自动计算)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细输出（当前版本默认不打印常规进度信息，仅在异常时输出）",
    )

    parser.add_argument(
        "--dry_run", action="store_true", help="试运行模式，只显示配置信息不执行计算"
    )

    return parser.parse_args()


def validate_files(h5_path, tsv_path):
    """
    验证输入文件

    Args:
        h5_path: HDF5文件路径
        tsv_path: TSV文件路径

    Returns:
        bool: 验证是否通过
    """
    # 检查文件存在性
    if not os.path.isfile(h5_path):
        print(f"错误: H5文件不存在: {h5_path}")
        return False

    if not os.path.isfile(tsv_path):
        print(f"错误: TSV文件不存在: {tsv_path}")
        return False

    # 检查H5文件内容
    try:
        with h5py.File(h5_path, "r") as h5f:
            required_datasets = ["SpecId", "Intpredict", "train_data", "Length"]
            missing_datasets = []

            for dataset in required_datasets:
                if dataset not in h5f:
                    missing_datasets.append(dataset)

            if missing_datasets:
                print(f"错误: H5文件缺少数据集: {missing_datasets}")
                return False

    except Exception as e:
        print(f"错误: 无法读取H5文件: {e}")
        return False

    # 检查TSV文件内容
    try:
        df = pd.read_csv(tsv_path, sep="\t", nrows=5)  # 只读前几行验证
        if "SpecId" not in df.columns:
            print("错误: TSV文件缺少 'SpecId' 列")
            return False

        # 获取完整行数（不打印统计信息）
        df_full = pd.read_csv(tsv_path, sep="\t", usecols=["SpecId"])

    except Exception as e:
        print(f"错误: 无法读取TSV文件: {e}")
        return False

    return True


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    # 验证输入文件
    if not validate_files(args.h5_path, args.tsv_path):
        sys.exit(1)

    if args.dry_run:
        # 静默模式：不打印额外信息
        sys.exit(0)

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        # 自动生成输出文件名
        workdir = os.path.dirname(os.path.abspath(args.tsv_path))
        if args.model_name == "mamba" or args.predictor_name == "Intpredict":
            output_path = os.path.join(workdir, "msms_specid_with_MS2PIP_m.tsv")
        else:
            output_path = os.path.join(
                workdir, f"{args.model_name}_msms_specid_with_MS2PIP_m.tsv"
            )

    # 如果输出路径与输入路径不同，确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 读入 TSV
        df = pd.read_csv(args.tsv_path, sep="\t")

        # 读入 H5（仅读取 SpecId 用于建立索引；具体强度矩阵在子进程内按需读取）
        with h5py.File(args.h5_path, "r") as h5f:
            spec_ids = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in h5f["SpecId"][:]
            ]

        # 构建索引映射
        sid2idx = {sid: i for i, sid in enumerate(spec_ids)}

        # 统计匹配/未匹配情况
        spec_ids_list = df["SpecId"].astype(str).tolist()
        missing = [sid for sid in spec_ids_list if sid not in sid2idx]
        matched_count = len(spec_ids_list) - len(missing)
        missing_count = len(missing)
        total_tsv = len(spec_ids_list)
        # 不打印匹配统计，保留潜在的警告信息

        if matched_count == 0:
            print("警告: 没有找到任何匹配的SpecId，请检查数据文件")

        # 多线程处理
        features_all = process_spectra_multithreaded(
            spec_ids_list,
            sid2idx,
            args.h5_path,
            args.workers,
            args.batch_size,
            args.predictor_name,
        )

        # 添加新列到DataFrame
        if features_all:
            # 获取所有特征名称
            feat_names = []
            for feats in features_all.values():
                if feats:  # 如果不是空字典
                    feat_names = list(feats.keys())
                    break

            if feat_names:
                # 添加特征列
                for feat in feat_names:
                    df[feat] = df["SpecId"].map(
                        lambda sid: features_all.get(sid, {}).get(feat, 0.0)
                    )
            else:
                print("警告: 没有计算出任何特征")
        else:
            print("警告: 特征计算结果为空")

        # 保存结果
        df.to_csv(output_path, sep="\t", index=False)
        # 静默完成，不打印总结信息

    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出。")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: 处理过程中发生异常: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
