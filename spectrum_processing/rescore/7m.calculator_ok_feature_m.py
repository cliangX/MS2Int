#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 PXD041529.h5 文件的特征计算工具（支持 203 维 m 离子）

输入格式: [npsm, 29, 31] -> 拼接为 [npsm, 203]
其中前7列为: b+, b++, y+, y++, m+, m++, m+3

使用方式示例:
    # 1) 仅基于 H5 计算特征（默认路径，输出不包含 Label/ScanNr/filename/Peptide/Proteins）
    python ok_feature/ok_feature_calculator.py \
        --h5_path /path/to/PXD041529.h5 \
        --output /path/to/features.tsv

    # 2) 仅基于 H5，打开详细日志
    python ok_feature/ok_feature_calculator.py \
        -i /path/to/PXD041529.h5 -o /path/to/features.tsv -v

    ###### 3) 基于 TSV 的 SpecId 对齐（左连接回原 TSV；未匹配行新增特征列为 NaN）
    python /mnt/data_nas/lcy/project_MS2predict/5.tools/mamba_rescore/7m.calculator_ok_feature_m.py \
        -i /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_MSV000087743/rescore/rescore.h5 \
        -t /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_MSV000087743/rescore/msms_specid.tsv \
        -o /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_MSV000087743/rescore/msms_specid_with_MS2PIP_m.tsv \
        -v

    # 4) 要求 TSV 的 SpecId 全部在 H5 中匹配（否则报错中止）
    python ok_feature/ok_feature_calculator.py \
        -i PXD041529.h5 -t input.tsv -o input_with_features.tsv --no-skip-missing

    # 5) 固定随机种子（如某些默认特征依赖随机数）
    python ok_feature/ok_feature_calculator.py \
        -i PXD041529.h5 -o features.tsv --seed 42 -v

注意:
    - H5 必需数据集: Intpredict [n,29,31], train_data [n,29,31], Length [n]
    - 若使用 TSV 对齐: H5 需包含数据集 SpecId，且 TSV 需包含列 SpecId
    - 输出的 TSV 默认不包含: Label, ScanNr, filename, Peptide, Proteins
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import pandas as pd

# 导入我们的特征计算器
# 优先从同目录的 ok_cpm.py 导入；若文件名为 9.ok_cpm.py，则走动态加载
from ok_cpm import CustomPercolatorFeatureCalculator  # 常规模块名


def load_h5_data(
    h5_path: str,
    predictor_name: str = "Intpredict",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    从H5文件加载数据

    Args:
        h5_path: H5文件路径
        predictor_name: 预测数据集名称（默认 Intpredict）

    Returns:
        Tuple: (intpred, train_data, lengths, spec_ids, has_specid)
    """
    logger = logging.getLogger(__name__)

    with h5py.File(h5_path, "r") as h5f:
        # logger.debug(f"H5文件包含的数据集: {list(h5f.keys())}")

        # 加载数据
        intpred = h5f[predictor_name][:]  # 预测强度 [npsm, 29, 31]
        train_data = h5f["train_data"][:]  # 观测强度 [npsm, 29, 31]
        lengths = h5f["Length"][:]  # 长度信息 [npsm]
        # SpecId（若存在）
        has_specid = False
        if "SpecId" in h5f:
            raw = h5f["SpecId"][:]
            try:
                spec_ids = np.array(
                    [
                        s.decode("utf-8")
                        if isinstance(s, (bytes, np.bytes_))
                        else str(s)
                        for s in raw
                    ]
                )
            except Exception:
                spec_ids = raw.astype(str)
            has_specid = True
        else:
            logger.warning("H5中未发现 'SpecId' 数据集，基于TSV的对齐将不可用")
            spec_ids = np.array([str(i) for i in range(len(lengths))])

        # logger.debug(f"Intpredict 形状: {intpred.shape}")
        # logger.debug(f"train_data 形状: {train_data.shape}")
        # logger.debug(f"Length 形状: {lengths.shape}")

        # 检查其他可能的数据集（静默模式）
        # for key in h5f.keys():
        #     if key not in ["Intpredict", "train_data", "Length"]:
        #         try:
        #             data = h5f[key][:]
        #             logger.debug(
        #                 f"{key} 形状: {data.shape if hasattr(data, 'shape') else type(data)}"
        #             )
        #         except:
        #             logger.debug(f"{key}: 无法读取")

    return intpred, train_data, lengths, spec_ids, has_specid


def reshape_matrix_to_116(matrix_3d: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    将 [npsm, 29, 31] 矩阵重塑为 [npsm, 116]

    提取前4列 (b+, b++, y+, y++) 并拼接为116维

    Args:
        matrix_3d: 输入矩阵 [npsm, 29, 31]
        lengths: 长度信息 [npsm]

    Returns:
        np.ndarray: 重塑后的矩阵 [npsm, 116]
    """
    npsm, seq_len, ion_types = matrix_3d.shape

    # 提取前4列: b+, b++, y+, y++
    selected_ions = matrix_3d[:, :, :4]  # [npsm, 29, 4]

    # 方法1: 简单拼接 - 将29x4=116维度直接展平
    matrix_116 = selected_ions.reshape(npsm, -1)  # [npsm, 116]

    logger = logging.getLogger(__name__)
    # logger.info(f"矩阵重塑: {matrix_3d.shape} -> {matrix_116.shape}")
    # logger.info(f"每个PSM使用前4种离子类型 (b+, b++, y+, y++) × 29个位置 = 116维")

    return matrix_116


def reshape_matrix_to_203(matrix_3d: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    将 [npsm, 29, 31] 矩阵重塑为 [npsm, 203]

    提取前7列 (b+, b++, y+, y++, m+, m++, m+3) 并拼接为203维

    Args:
        matrix_3d: 输入矩阵 [npsm, 29, 31]
        lengths: 长度信息 [npsm]

    Returns:
        np.ndarray: 重塑后的矩阵 [npsm, 203]
    """
    npsm, seq_len, ion_types = matrix_3d.shape

    # 提取前7列: b+, b++, y+, y++, m+, m++, m+3
    selected_ions = matrix_3d[:, :, :7]  # [npsm, 29, 7]

    # 简单拼接 - 将29x7=203维度直接展平
    matrix_203 = selected_ions.reshape(npsm, -1)  # [npsm, 203]

    logger = logging.getLogger(__name__)
    logger.info(f"矩阵重塑: {matrix_3d.shape} -> {matrix_203.shape}")
    logger.info(
        "每个PSM使用前7种离子类型 (b+, b++, y+, y++, m+, m++, m+3) × 29个位置 = 203维"
    )

    return matrix_203


def create_metadata_from_h5(h5_path: str, npsm: int) -> pd.DataFrame:
    """Deprecated: 元数据构造逻辑按需求移除。

    如需进行基于序列/电荷等的特征，请改为提供 TSV 并在其中包含相关列，
    或在调用处以空 DataFrame 占位（将自动使用默认特征）。
    """
    raise NotImplementedError(
        "create_metadata_from_h5 has been removed. Provide TSV metadata or use empty DataFrame."
    )


def analyze_ion_distribution(
    matrix_3d: np.ndarray, lengths: np.ndarray
) -> Dict[str, Any]:
    """
    分析离子强度分布

    Args:
        matrix_3d: 输入矩阵 [npsm, 29, 31]
        lengths: 长度信息 [npsm]

    Returns:
        Dict: 分析结果
    """
    logger = logging.getLogger(__name__)

    # 提取前7列（含 m 离子）
    ion_matrix = matrix_3d[:, :, :7]  # [npsm, 29, 7]

    ion_names = ["b+", "b++", "y+", "y++", "m+", "m++", "m+3"]
    analysis = {
        "ion_names": ion_names,
        "matrix_shape": ion_matrix.shape,
        "total_intensities": {},
        "non_zero_counts": {},
        "mean_intensities": {},
        "intensity_ranges": {},
    }

    for i, ion_name in enumerate(ion_names):
        ion_data = ion_matrix[:, :, i]

        analysis["total_intensities"][ion_name] = float(np.sum(ion_data))
        analysis["non_zero_counts"][ion_name] = int(np.sum(ion_data > 0))
        analysis["mean_intensities"][ion_name] = float(np.mean(ion_data))
        analysis["intensity_ranges"][ion_name] = {
            "min": float(np.min(ion_data)),
            "max": float(np.max(ion_data)),
            "std": float(np.std(ion_data)),
        }

    # 整体统计
    analysis["overall"] = {
        "total_elements": int(ion_matrix.size),
        "non_zero_elements": int(np.sum(ion_matrix > 0)),
        "sparsity": float(1 - np.sum(ion_matrix > 0) / ion_matrix.size),
        "mean_intensity": float(np.mean(ion_matrix)),
        "std_intensity": float(np.std(ion_matrix)),
    }

    # 长度统计
    analysis["lengths"] = {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }

    logger.info("离子分布分析完成")
    for ion_name in ion_names:
        logger.info(
            f"{ion_name}: 非零元素={analysis['non_zero_counts'][ion_name]}, 平均强度={analysis['mean_intensities'][ion_name]:.4f}"
        )

    return analysis


def _merge_features_to_tsv(
    tsv_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    how: str = "left",
    fill_missing: str = "nan",
) -> pd.DataFrame:
    """
    将计算出的特征按 SpecId 回并到 TSV，保持 TSV 原始行序

    Args:
        tsv_df: 原始TSV数据
        feature_df: 只包含匹配行的特征数据（需包含SpecId列）
        how: 合并方式（默认left）
        fill_missing: 未匹配行的填充策略："nan" 或 "zero"
    """
    if "SpecId" not in tsv_df.columns:
        raise ValueError("TSV缺少必需列 'SpecId'")
    if "SpecId" not in feature_df.columns:
        raise ValueError("特征结果缺少必需列 'SpecId'")

    merged = tsv_df.merge(feature_df, on="SpecId", how=how)

    if fill_missing == "zero":
        # 仅填充新增的特征列
        orig_cols = set(tsv_df.columns)
        new_cols = [c for c in merged.columns if c not in orig_cols]
        for c in new_cols:
            if pd.api.types.is_numeric_dtype(merged[c]):
                merged[c] = merged[c].fillna(0.0)
    return merged


def calculate_features_from_h5(
    h5_path: str,
    output_path: str,
    verbose: bool = False,
    tsv_path: str = None,
    skip_missing: bool = True,
    seed: int = None,
    fill_missing: str = "nan",
    predictor_name: str = "Intpredict",
):
    """
    从H5文件计算特征（203维 m 离子版本）

    Args:
        h5_path: H5文件路径
        output_path: 输出文件路径
        verbose: 是否详细输出
        tsv_path: TSV路径（若提供则按TSV的SpecId顺序对齐，并左连接回TSV）
        skip_missing: 跳过未匹配的SpecId参与计算（默认True），但依然在输出中保留行
        seed: 随机种子（若后续需要随机行为也可固定）
        fill_missing: 未匹配行特征填充策略："nan"（默认）或 "zero"
        predictor_name: 预测数据集名称（默认 Intpredict）
    """
    logger = logging.getLogger(__name__)
    if seed is not None:
        try:
            np.random.seed(int(seed))
        except Exception:
            logger.warning(f"无法设置随机种子: {seed}")
    else:
        # 默认设置固定种子以确保结果可重复
        np.random.seed(42)

    # 1. 加载数据
    logger.info("=" * 60)
    logger.info("开始处理 数据（203维 m 离子版本）")
    logger.info("=" * 60)

    intpred, train_data, lengths, spec_ids, has_specid = load_h5_data(
        h5_path, predictor_name
    )
    npsm = intpred.shape[0]

    logger.info(f"成功加载数据: {npsm} 个PSM (predictor={predictor_name})")

    # 若未提供TSV，沿用原有全量计算逻辑
    if not tsv_path:
        # 2. 分析数据分布
        if verbose:
            logger.info("\n分析预测数据分布...")
            pred_analysis = analyze_ion_distribution(intpred, lengths)

            logger.info("\n分析观测数据分布...")
            obs_analysis = analyze_ion_distribution(train_data, lengths)

            # 打印关键统计信息
            logger.info(
                f"\n数据稀疏性: 预测={pred_analysis['overall']['sparsity']:.1%}, 观测={obs_analysis['overall']['sparsity']:.1%}"
            )

        # 3. 重塑矩阵到 203 维
        logger.info("\n重塑矩阵到203维...")
        pred_matrix_203 = reshape_matrix_to_203(intpred, lengths)
        obs_matrix_203 = reshape_matrix_to_203(train_data, lengths)

        logger.info(f"重塑完成: {pred_matrix_203.shape}, {obs_matrix_203.shape}")

        # 4. 跳过元数据构造：按需求移除，使用空元数据（特征计算器将使用默认值）
        logger.info(
            "\n跳过元数据构造，使用空元数据（将使用默认序列/电荷等默认特征）..."
        )
        metadata = pd.DataFrame(index=np.arange(npsm))

        # 5. 计算特征
        logger.info("\n初始化特征计算器（203维）...")

        calculator = CustomPercolatorFeatureCalculator(
            obs_matrix=obs_matrix_203,
            pred_matrix=pred_matrix_203,
            metadata=metadata,
            fragment_dim=203,
        )

        logger.info("开始特征计算...")
        feature_df = calculator.calculate_all_features()

        # 6. 按需求移除不需要的列（仅影响最终输出，不影响内部计算）
        # 需求：输出的TSV不需要包含以下列
        drop_cols = ["Label", "ScanNr", "filename", "Peptide", "Proteins"]
        existing_drop = [c for c in drop_cols if c in feature_df.columns]
        if existing_drop:
            logger.info(f"移除输出中不需要的列: {existing_drop}")
            feature_df = feature_df.drop(columns=existing_drop)

        # 7. 保存结果
        logger.info(f"保存结果到: {output_path}")
        feature_df.to_csv(output_path, sep="\t", index=False)

        # 8. 显示结果统计
        logger.info("\n" + "=" * 60)
        logger.info("特征计算完成！")
        logger.info("=" * 60)

        logger.info(f"输入PSM数量: {npsm}")
        logger.info("输入矩阵维度: [npsm, 29, 31] -> [npsm, 203]")
        logger.info(f"输出文件: {output_path}")

        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"文件大小: {file_size_mb:.2f} MB")
        return

    # 提供TSV：按TSV的SpecId顺序对齐
    logger.info(f"\n读取TSV: {tsv_path}")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV文件不存在: {tsv_path}")
    tsv_df = pd.read_csv(tsv_path, sep="\t")
    if "SpecId" not in tsv_df.columns:
        raise ValueError("TSV文件缺少 'SpecId' 列")

    if not has_specid:
        raise ValueError(
            "H5文件缺少 'SpecId' 数据集，无法执行基于TSV的对齐。请先运行添加SpecId的数据处理步骤。"
        )

    # 构建SpecId->索引映射
    sid2idx = {sid: i for i, sid in enumerate(spec_ids)}
    tsv_specids = tsv_df["SpecId"].astype(str).tolist()

    # 按 TSV 行顺序一一匹配，记录对应的 H5 索引和 TSV 行索引
    matched_index = []  # TSV 行索引列表（保持原行序，支持重复 SpecId）
    sel_indices = []  # H5 中对应的行索引（可重复）
    for row_idx, sid in zip(tsv_df.index.tolist(), tsv_specids):
        if sid in sid2idx:
            matched_index.append(row_idx)
            sel_indices.append(sid2idx[sid])

    matched_count = len(sel_indices)
    missing_count = len(tsv_specids) - matched_count
    logger.info(
        f"TSV匹配: {matched_count}/{len(tsv_specids)} 在H5中找到；未匹配 {missing_count}"
    )
    if verbose and missing_count > 0:
        missing_examples = [sid for sid in tsv_specids if sid not in sid2idx][:10]
        logger.info(f"未匹配示例(最多10): {missing_examples}")

    if (not skip_missing) and missing_count > 0:
        raise ValueError(
            f"存在未匹配的SpecId数量: {missing_count}（共{len(tsv_specids)}）。"
            f"当前 --skip_missing 被禁用，已中止以避免输出缺失特征。"
        )

    if matched_count == 0:
        logger.warning("没有任何TSV中的SpecId在H5中匹配，直接原样输出TSV")
        tsv_df.to_csv(output_path, sep="\t", index=False)
        return

    # 保持TSV行序一一提取匹配到的行（避免基于 SpecId 的多对多合并）
    tsv_matched_metadata = tsv_df.loc[matched_index].copy()

    # 重塑矩阵到 203 维（不做长度屏蔽处理）
    # logger.debug("\n重塑矩阵到203维（保持TSV顺序）...")
    pred_matrix_203 = reshape_matrix_to_203(
        intpred[sel_indices, ...], lengths[sel_indices]
    )
    obs_matrix_203 = reshape_matrix_to_203(
        train_data[sel_indices, ...], lengths[sel_indices]
    )
    # logger.debug(f"重塑完成: {pred_matrix_203.shape}, {obs_matrix_203.shape}")

    # 初始化特征计算
    # logger.debug("\n初始化特征计算器（203维）...")

    # 元数据直接使用TSV匹配到的行（至少包含SpecId）
    metadata = tsv_matched_metadata.reset_index(drop=True)

    calculator = CustomPercolatorFeatureCalculator(
        obs_matrix=obs_matrix_203,
        pred_matrix=pred_matrix_203,
        metadata=metadata,
        fragment_dim=203,
    )

    logger.info("开始特征计算...")
    feature_df = calculator.calculate_all_features()
    if "SpecId" not in feature_df.columns and "SpecId" in metadata.columns:
        # 可选：保持 SpecId 便于调试；回并时按索引进行，避免多对多
        feature_df.insert(0, "SpecId", metadata["SpecId"].values)

    # 按需求移除不需要的列（仅影响最终输出，不影响内部计算）
    drop_cols = ["Label", "ScanNr", "filename", "Peptide", "Proteins"]
    existing_drop = [c for c in drop_cols if c in feature_df.columns]
    if existing_drop:
        # logger.debug(f"移除输出中不需要的列: {existing_drop}")
        feature_df = feature_df.drop(columns=existing_drop)

    # 关键改动：按 TSV 行索引一一回并，避免 SpecId 重复导致的多对多合并行数膨胀
    # 将特征行的索引设置为其对应的原 TSV 行索引
    feature_df.index = pd.Index(matched_index)

    # 回并前去掉会与 TSV 重复的标识列，避免产生后缀列
    cols_to_drop_before_join = [
        c
        for c in ["SpecId", "Label", "ScanNr", "filename", "Peptide", "Proteins"]
        if c in feature_df.columns
    ]
    if cols_to_drop_before_join:
        feature_cols_kept = [
            c for c in feature_df.columns if c not in cols_to_drop_before_join
        ]
        feature_df = feature_df[feature_cols_kept]

    # 避免与 TSV 中现有列名冲突，将重叠特征列加上 ok_ 前缀
    overlap_feature_cols = [c for c in feature_df.columns if c in tsv_df.columns]
    if overlap_feature_cols:
        rename_map = {c: f"ok_{c}" for c in overlap_feature_cols}
        feature_df = feature_df.rename(columns=rename_map)

    # logger.debug("\n按索引左连接特征回 TSV（保持行序，一对一回并）...")
    merged_df = tsv_df.join(feature_df, how="left")

    # 同步生成"仅保留 SpecId + 特征"的版本
    features_only_df = tsv_df[["SpecId"]].join(feature_df, how="left")

    # 未匹配行的特征填充策略
    if fill_missing == "zero":
        new_cols = [c for c in feature_df.columns]
        for c in new_cols:
            if pd.api.types.is_numeric_dtype(merged_df[c]):
                merged_df[c] = merged_df[c].fillna(0.0)
            if pd.api.types.is_numeric_dtype(features_only_df[c]):
                features_only_df[c] = features_only_df[c].fillna(0.0)

    # 计算两个输出文件路径（与 TSV 同目录）
    base_dir = os.path.dirname(os.path.abspath(tsv_path))
    out_merged = os.path.join(base_dir, "msms_specid_with_ms2pip_ok_m.tsv")
    out_features_only = os.path.join(base_dir, "msms_specid_with_ok_m.tsv")

    # 还保留原先的 --output 路径（向后兼容）
    logger.info(f"保存合并版到: {out_merged}")
    merged_df.to_csv(out_merged, sep="\t", index=False)
    logger.info(f"保存特征版到: {out_features_only}")
    features_only_df.to_csv(out_features_only, sep="\t", index=False)

    # 兼容性输出
    try:
        logger.info(f"另存为(兼容旧参数 --output): {output_path}")
        merged_df.to_csv(output_path, sep="\t", index=False)
    except Exception as e:
        logger.warning(f"写入 --output 路径失败（忽略）：{e}")

    logger.info(
        "完成：已输出 msms_specid_with_ms2pip_ok_m.tsv 与 msms_specid_with_ok_m.tsv 两个版本"
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="处理 PXD041529.h5 文件的特征计算工具（203维 m 离子版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 仅基于 H5 计算特征
  %(prog)s --h5_path PXD041529.h5 --output features.tsv

  # 基于 TSV 的 SpecId 对齐（左连接），保持 TSV 行序
  %(prog)s --h5_path PXD041529.h5 --tsv_path input.tsv --output input_with_features.tsv

  # 打开详细日志/固定种子/严格匹配
  %(prog)s -i PXD041529.h5 -o features.tsv -v --seed 42
  %(prog)s -i PXD041529.h5 -t input.tsv -o out.tsv --no-skip-missing

数据格式说明:
  - 输入 H5: [n, 29, 31]，仅取前7个离子通道 (b+, b++, y+, y++, m+, m++, m+3) → 展平为 203 维
  - 若 TSV 对齐: 需要 H5 含数据集 SpecId，TSV 含列 SpecId
  - 输出 TSV 默认不含列: Label, ScanNr, filename, Peptide, Proteins
        """,
    )

    parser.add_argument(
        "--h5_path", "-i", type=str, required=True, help=".h5 文件路径 (必需)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出特征文件路径 (可选，默认自动生成)",
    )
    parser.add_argument(
        "--tsv_path",
        "-t",
        type=str,
        default=None,
        help="若提供，则按TSV的SpecId顺序对齐并将特征左连接回该TSV",
    )

    parser.add_argument(
        "--predictor_name",
        type=str,
        default="Intpredict",
        help="h5文件中预测结果的数据集名称（默认 Intpredict，可指定 AlphaPeptDeep/Prosit_HCD 等）",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="mamba",
        help="模型名称，用于自动生成输出文件名（默认 mamba）",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示详细输出和数据分析"
    )
    parser.add_argument(
        "--skip_missing",
        dest="skip_missing",
        action="store_true",
        default=True,
        help="跳过未匹配的SpecId参与计算，但仍在输出中保留原始行（左连接）（默认启用）",
    )
    parser.add_argument(
        "--no-skip-missing",
        dest="skip_missing",
        action="store_false",
        help="若存在未匹配的SpecId则报错并中止（与 --skip_missing 相反）",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（如需固定随机行为）"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)

    # 检查输入文件
    if not os.path.isfile(args.h5_path):
        logger.error(f"H5文件不存在: {args.h5_path}")
        sys.exit(1)

    # 确定输出路径（自动生成文件名）
    if args.output:
        output_path = args.output
    else:
        # 自动生成输出文件名
        if args.tsv_path:
            workdir = os.path.dirname(os.path.abspath(args.tsv_path))
        else:
            workdir = os.path.dirname(os.path.abspath(args.h5_path))

        if args.model_name == "mamba" or args.predictor_name == "Intpredict":
            output_path = os.path.join(workdir, "msms_specid_with_ok_m.tsv")
        else:
            output_path = os.path.join(
                workdir, f"{args.model_name}_msms_specid_with_ok_m.tsv"
            )

        logger.info(f"自动生成输出路径: {output_path}")

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    try:
        # 执行特征计算
        calculate_features_from_h5(
            h5_path=args.h5_path,
            output_path=output_path,
            verbose=args.verbose,
            tsv_path=args.tsv_path,
            skip_missing=args.skip_missing,
            seed=args.seed,
            fill_missing="nan",  # 默认未匹配行留空
            predictor_name=args.predictor_name,
        )

    except Exception as e:
        logger.error(f"处理失败: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
