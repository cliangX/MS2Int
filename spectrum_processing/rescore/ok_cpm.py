#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义谱图特征计算工具 - 支持 (N_PSM, 116) 格式输入

参考 Percolator 特征计算算法，适配自定义的数据格式
支持观测强度矩阵和预测强度矩阵的特征计算

使用方式:
    python ok_cpm.py \
        --obs_matrix_path /path/to/observed_intensities.npy \
        --pred_matrix_path /path/to/predicted_intensities.npy \
        --metadata_path /path/to/metadata.csv \
        --output /path/to/features.tab \
        --verbose

输入格式要求:
    obs_matrix: 观测强度矩阵 (N_PSM, 116) - numpy array 格式
    pred_matrix: 预测强度矩阵 (N_PSM, 116) - numpy array 格式  
    metadata: PSM元数据 (CSV格式) - 包含必要的肽段信息
"""

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# 默认关闭 numba 的调试/诊断输出，避免污染日志
_NUMBA_DEBUG_ENV_VARS = (
    "NUMBA_DEBUG",
    "NUMBA_DEBUG_ARRAY_OPT",
    "NUMBA_DEBUG_CACHE",
    "NUMBA_DEBUG_DISPATCH",
    "NUMBA_DEBUG_JIT",
    "NUMBA_DEBUG_LLVM",
    "NUMBA_DEBUG_NRT",
    "NUMBA_DEBUG_TYPEINFER",
    "NUMBA_DUMP_CFG",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_PYCFG",
    "NUMBA_DUMP_TYPEINFER",
    "NUMBA_PARALLEL_DIAGNOSTICS",
)
for _env_var in _NUMBA_DEBUG_ENV_VARS:
    if os.environ.get(_env_var):
        os.environ.pop(_env_var)

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback when numba missing
    NUMBA_AVAILABLE = False
    njit = prange = None  # type: ignore

NUMERIC_EPS = 1e-12


def _prepare_numba_inputs(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Ensure arrays are C-contiguous float64 for numba kernels."""
    prepared: List[np.ndarray] = []
    for arr in arrays:
        if arr.dtype != np.float64 or not arr.flags.c_contiguous:
            prepared.append(np.ascontiguousarray(arr, dtype=np.float64))
        else:
            prepared.append(arr)
    return tuple(prepared)


if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)
    def _spectral_angle_kernel(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        n_rows, n_cols = obs.shape
        out = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            dot = 0.0
            norm_obs = 0.0
            norm_pred = 0.0
            for j in range(n_cols):
                x = obs[i, j]
                y = pred[i, j]
                dot += x * y
                norm_obs += x * x
                norm_pred += y * y
            norm_obs = math.sqrt(norm_obs)
            norm_pred = math.sqrt(norm_pred)
            denom = norm_obs * norm_pred
            if denom <= NUMERIC_EPS:
                out[i] = 0.0
            else:
                cos_val = dot / denom
                if cos_val < -1.0:
                    cos_val = -1.0
                elif cos_val > 1.0:
                    cos_val = 1.0
                angle = math.acos(cos_val)
                out[i] = 1.0 - 2.0 * angle / math.pi
        return out

    @njit(parallel=True, fastmath=True)
    def _cosine_similarity_kernel(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        n_rows, n_cols = obs.shape
        out = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            dot = 0.0
            norm_obs = 0.0
            norm_pred = 0.0
            for j in range(n_cols):
                x = obs[i, j]
                y = pred[i, j]
                dot += x * y
                norm_obs += x * x
                norm_pred += y * y
            norm_obs = math.sqrt(norm_obs)
            norm_pred = math.sqrt(norm_pred)
            denom = norm_obs * norm_pred
            if denom <= NUMERIC_EPS:
                out[i] = 0.0
            else:
                val = dot / denom
                if val < -1.0:
                    val = -1.0
                elif val > 1.0:
                    val = 1.0
                out[i] = val
        return out

    @njit(fastmath=True)
    def _rankdata(values: np.ndarray) -> np.ndarray:
        n = values.shape[0]
        ranks = np.empty(n, dtype=np.float64)
        order = np.argsort(values)
        i = 0
        while i < n:
            start = i
            current = values[order[i]]
            while i + 1 < n and values[order[i + 1]] == current:
                i += 1
            end = i
            rank = (start + end) * 0.5 + 1.0
            for j in range(start, end + 1):
                ranks[order[j]] = rank
            i += 1
        return ranks

    @njit(parallel=True, fastmath=True)
    def _pearson_correlation_kernel(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        n_rows, n_cols = obs.shape
        out = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            mean_obs = 0.0
            mean_pred = 0.0
            for j in range(n_cols):
                mean_obs += obs[i, j]
                mean_pred += pred[i, j]
            mean_obs /= n_cols
            mean_pred /= n_cols

            num = 0.0
            denom_obs = 0.0
            denom_pred = 0.0
            for j in range(n_cols):
                dx = obs[i, j] - mean_obs
                dy = pred[i, j] - mean_pred
                num += dx * dy
                denom_obs += dx * dx
                denom_pred += dy * dy
            denom = math.sqrt(denom_obs * denom_pred)
            if denom <= NUMERIC_EPS:
                out[i] = 0.0
            else:
                val = num / denom
                if val < -1.0:
                    val = -1.0
                elif val > 1.0:
                    val = 1.0
                out[i] = val
        return out

    @njit(parallel=True, fastmath=True)
    def _spearman_correlation_kernel(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        n_rows, _ = obs.shape
        out = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            ranks_obs = _rankdata(obs[i])
            ranks_pred = _rankdata(pred[i])

            mean_obs = 0.0
            mean_pred = 0.0
            n_cols = ranks_obs.shape[0]
            for j in range(n_cols):
                mean_obs += ranks_obs[j]
                mean_pred += ranks_pred[j]
            mean_obs /= n_cols
            mean_pred /= n_cols

            num = 0.0
            denom_obs = 0.0
            denom_pred = 0.0
            for j in range(n_cols):
                dx = ranks_obs[j] - mean_obs
                dy = ranks_pred[j] - mean_pred
                num += dx * dy
                denom_obs += dx * dx
                denom_pred += dy * dy
            denom = math.sqrt(denom_obs * denom_pred)
            if denom <= NUMERIC_EPS:
                out[i] = 0.0
            else:
                val = num / denom
                if val < -1.0:
                    val = -1.0
                elif val > 1.0:
                    val = 1.0
                out[i] = val
        return out

    def _spectral_angle_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        obs_prepped, pred_prepped = _prepare_numba_inputs(obs, pred)
        return _spectral_angle_kernel(obs_prepped, pred_prepped)

    def _cosine_similarity_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        obs_prepped, pred_prepped = _prepare_numba_inputs(obs, pred)
        return _cosine_similarity_kernel(obs_prepped, pred_prepped)

    def _pearson_correlation_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        obs_prepped, pred_prepped = _prepare_numba_inputs(obs, pred)
        return _pearson_correlation_kernel(obs_prepped, pred_prepped)

    def _spearman_correlation_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        obs_prepped, pred_prepped = _prepare_numba_inputs(obs, pred)
        return _spearman_correlation_kernel(obs_prepped, pred_prepped)

else:

    def _spectral_angle_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        obs_norm = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + NUMERIC_EPS)
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + NUMERIC_EPS)
        dot_product = np.sum(obs_norm * pred_norm, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        spectral_angle = 1 - 2 * angle / np.pi
        return np.nan_to_num(spectral_angle)

    def _cosine_similarity_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        dot_product = np.sum(obs * pred, axis=1)
        obs_norm = np.linalg.norm(obs, axis=1)
        pred_norm = np.linalg.norm(pred, axis=1)
        cosine_sim = dot_product / (obs_norm * pred_norm + NUMERIC_EPS)
        return np.nan_to_num(cosine_sim)

    def _pearson_correlation_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        correlations = []
        for i in range(obs.shape[0]):
            obs_row = obs[i]
            pred_row = pred[i]
            if np.std(obs_row) <= NUMERIC_EPS or np.std(pred_row) <= NUMERIC_EPS:
                correlations.append(0.0)
            else:
                corr = np.corrcoef(obs_row, pred_row)[0, 1]
                correlations.append(np.nan_to_num(corr))
        return np.array(correlations)

    def _spearman_correlation_impl(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        from scipy import stats

        correlations = []
        for i in range(obs.shape[0]):
            obs_row = obs[i]
            pred_row = pred[i]
            try:
                corr, _ = stats.spearmanr(obs_row, pred_row)
                correlations.append(np.nan_to_num(corr))
            except Exception:
                correlations.append(0.0)
        return np.array(correlations)


class CustomPercolatorFeatureCalculator:
    """
    自定义 Percolator 特征计算器

    适配 (N_PSM, 116) 格式的输入数据，计算基于 Percolator 算法的谱图匹配特征
    """

    def __init__(
        self,
        obs_matrix: np.ndarray,
        pred_matrix: np.ndarray,
        metadata: pd.DataFrame,
        fragment_dim: int = 116,
    ):
        """
        初始化特征计算器

        Args:
            obs_matrix: 观测强度矩阵 (N_PSM, 116)
            pred_matrix: 预测强度矩阵 (N_PSM, 116)
            metadata: 元数据DataFrame
            fragment_dim: 碎片维度 (默认116)
        """
        self.obs_matrix = obs_matrix
        self.pred_matrix = pred_matrix
        self.metadata = metadata
        self.fragment_dim = fragment_dim
        self.n_psm = obs_matrix.shape[0]

        # 先初始化 logger，避免 _validate_input 使用时不存在
        self.logger = logging.getLogger(self.__class__.__name__)

        # 验证输入
        self._validate_input()

        # 创建质荷比矩阵 (如果没有提供)
        self.mz_matrix = self._create_dummy_mz_matrix()

        # 初始化特征存储
        self.features = {}

    def _validate_input(self):
        """验证输入数据的有效性"""
        # 检查矩阵维度
        if self.obs_matrix.shape != self.pred_matrix.shape:
            raise ValueError(
                f"观测矩阵 {self.obs_matrix.shape} 与预测矩阵 {self.pred_matrix.shape} 形状不匹配"
            )

        if self.obs_matrix.shape[1] != self.fragment_dim:
            raise ValueError(
                f"期望碎片维度 {self.fragment_dim}，实际得到 {self.obs_matrix.shape[1]}"
            )

        if len(self.metadata) != self.n_psm:
            raise ValueError(
                f"元数据行数 {len(self.metadata)} 与PSM数量 {self.n_psm} 不匹配"
            )

        # 检查必要列（静默模式）
        required_columns = ["SEQUENCE", "PRECURSOR_CHARGE", "REVERSE"]
        missing_columns = [
            col for col in required_columns if col not in self.metadata.columns
        ]
        # if missing_columns:
        #     self.logger.warning(f"元数据缺少推荐列: {missing_columns}")

    def _create_dummy_mz_matrix(self) -> np.ndarray:
        """创建虚拟质荷比矩阵"""
        # 为116维创建合理的质荷比值
        # 假设质荷比范围在 200-2000 之间
        base_mz = np.linspace(200, 2000, self.fragment_dim)
        mz_matrix = np.tile(base_mz, (self.n_psm, 1))

        # 添加一些随机变化以模拟真实数据
        mz_matrix += np.random.normal(0, 10, mz_matrix.shape)
        mz_matrix = np.abs(mz_matrix)  # 确保质荷比为正值

        return mz_matrix

    def _adapt_matrix_to_174(self, matrix_116: np.ndarray) -> np.ndarray:
        """
        将116维矩阵适配到174维标准格式

        方法：
        1. 零填充: 直接在末尾填充零至174维
        2. 重复映射: 将116维按比例映射到174维
        3. 线性插值: 使用插值方法扩展到174维

        Args:
            matrix_116: 输入的116维矩阵 (N_PSM, 116)

        Returns:
            np.ndarray: 174维矩阵 (N_PSM, 174)
        """
        n_psm = matrix_116.shape[0]

        # 方法1: 零填充 (最简单)
        if self.fragment_dim <= 174:
            matrix_174 = np.zeros((n_psm, 174))
            matrix_174[:, : self.fragment_dim] = matrix_116
            return matrix_174

        # 方法2: 截断 (如果输入维度大于174)
        else:
            return matrix_116[:, :174]

    def _create_ion_masks_116(self) -> Dict[str, np.ndarray]:
        """
        通用离子掩码生成（支持 116 和 203 维）

        布局假设：展平矩阵按 [位置0的所有通道, 位置1的所有通道, ...] 排列
        - 116 维: 29 位置 × 4 通道 [b+, b++, y+, y++]
        - 203 维: 29 位置 × 7 通道 [b+, b++, y+, y++, m+, m++, m+3]

        Returns:
            Dict: 包含不同离子类型掩码的字典
        """
        dim = int(self.fragment_dim)

        # 推断每位置的通道数
        if dim == 116:
            n_types = 4  # [b+, b++, y+, y++]
        elif dim == 203:
            n_types = 7  # [b+, b++, y+, y++, m+, m++, m+3]
        elif dim % 29 == 0:
            n_types = dim // 29  # 通用推断
        else:
            raise ValueError(
                f"无法从 fragment_dim={dim} 推断通道数。"
                f"当前仅支持 116 维（4通道）和 203 维（7通道）"
            )

        # 计算每个索引对应的通道号（核心：取模运算）
        idx = np.arange(dim)
        ion_idx = idx % n_types

        # 定义通道映射（与 reshape 时的列顺序保持一致）
        if n_types == 4:
            b_channels = {0, 1}
            y_channels = {2, 3}
            m_channels = set()
        elif n_types == 7:
            b_channels = {0, 1}
            y_channels = {2, 3}
            m_channels = {4, 5, 6}
        else:
            # 通用默认：前半 b，后半 y
            b_channels = set(range(n_types // 2))
            y_channels = set(range(n_types // 2, n_types))
            m_channels = set()

        # 使用 numpy.isin 生成掩码
        masks = {
            "y_ion_mask": np.isin(ion_idx, list(y_channels)).astype(float),
            "b_ion_mask": np.isin(ion_idx, list(b_channels)).astype(float),
            "all_mask": np.ones(dim, dtype=float),
            "first_third": np.concatenate(
                [
                    np.ones(dim // 3),
                    np.zeros(dim - dim // 3),
                ]
            ),
            "middle_third": np.concatenate(
                [
                    np.zeros(dim // 3),
                    np.ones(dim // 3),
                    np.zeros(dim - 2 * (dim // 3)),
                ]
            ),
            "last_third": np.concatenate(
                [
                    np.zeros(2 * (dim // 3)),
                    np.ones(dim - 2 * (dim // 3)),
                ]
            ),
        }

        # 可选：添加 m 离子掩码（203 维时可用）
        if m_channels:
            masks["m_ion_mask"] = np.isin(ion_idx, list(m_channels)).astype(float)

        return masks

    def calculate_basic_similarity_features(self) -> Dict[str, np.ndarray]:
        """
        计算基础相似性特征

        Returns:
            Dict: 相似性特征字典
        """
        features = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 1. 光谱角 (Spectral Angle)
            features["spectral_angle"] = self._calculate_spectral_angle(
                self.obs_matrix, self.pred_matrix
            )

            # 2. 皮尔逊相关系数
            features["pearson_corr"] = self._calculate_pearson_correlation(
                self.obs_matrix, self.pred_matrix
            )

            # 3. 斯皮尔曼相关系数
            features["spearman_corr"] = self._calculate_spearman_correlation(
                self.obs_matrix, self.pred_matrix
            )

            # 4. 余弦相似度
            features["cosine_sim"] = self._calculate_cosine_similarity(
                self.obs_matrix, self.pred_matrix
            )

            # 5. 按离子类型分别计算
            ion_masks = self._create_ion_masks_116()

            for mask_name, mask in ion_masks.items():
                if mask_name in ["y_ion_mask", "b_ion_mask"]:
                    ion_type = mask_name.split("_")[0]  # 'y' or 'b'

                    obs_masked = self.obs_matrix * mask[np.newaxis, :]
                    pred_masked = self.pred_matrix * mask[np.newaxis, :]

                    features[f"spectral_angle_{ion_type}"] = (
                        self._calculate_spectral_angle(obs_masked, pred_masked)
                    )
                    features[f"pearson_corr_{ion_type}"] = (
                        self._calculate_pearson_correlation(obs_masked, pred_masked)
                    )
                    features[f"cosine_sim_{ion_type}"] = (
                        self._calculate_cosine_similarity(obs_masked, pred_masked)
                    )

        return features

    def calculate_fragments_statistics(
        self, threshold: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        计算碎片离子统计特征

        Args:
            threshold: 强度阈值，用于判断离子是否存在

        Returns:
            Dict: 碎片统计特征字典
        """
        features = {}

        # 创建布尔掩码
        obs_mask = self.obs_matrix > threshold
        pred_mask = self.pred_matrix > threshold

        # 基础计数
        features["count_observed"] = np.sum(obs_mask, axis=1).astype(float)
        features["count_predicted"] = np.sum(pred_mask, axis=1).astype(float)
        features["count_observed_and_predicted"] = np.sum(
            obs_mask & pred_mask, axis=1
        ).astype(float)
        features["count_observed_not_predicted"] = np.sum(
            obs_mask & ~pred_mask, axis=1
        ).astype(float)
        features["count_predicted_not_observed"] = np.sum(
            ~obs_mask & pred_mask, axis=1
        ).astype(float)

        # 比例特征
        total_fragments = float(self.fragment_dim)
        features["fraction_observed"] = features["count_observed"] / total_fragments
        features["fraction_predicted"] = features["count_predicted"] / total_fragments
        features["fraction_observed_and_predicted"] = (
            features["count_observed_and_predicted"] / total_fragments
        )

        # 匹配效率
        features["fraction_matched"] = features[
            "count_observed_and_predicted"
        ] / np.maximum(features["count_predicted"], 1.0)
        features["fraction_recall"] = features[
            "count_observed_and_predicted"
        ] / np.maximum(features["count_observed"], 1.0)

        # 按离子类型统计
        ion_masks = self._create_ion_masks_116()

        for mask_name, mask in ion_masks.items():
            if mask_name in ["y_ion_mask", "b_ion_mask"]:
                ion_type = mask_name.split("_")[0]  # 'y' or 'b'

                obs_masked = obs_mask & mask[np.newaxis, :].astype(bool)
                pred_masked = pred_mask & mask[np.newaxis, :].astype(bool)

                features[f"count_observed_{ion_type}"] = np.sum(
                    obs_masked, axis=1
                ).astype(float)
                features[f"count_predicted_{ion_type}"] = np.sum(
                    pred_masked, axis=1
                ).astype(float)
                features[f"count_observed_and_predicted_{ion_type}"] = np.sum(
                    obs_masked & pred_masked, axis=1
                ).astype(float)

                mask_total = np.sum(mask)
                features[f"fraction_observed_{ion_type}"] = (
                    features[f"count_observed_{ion_type}"] / mask_total
                )
                features[f"fraction_predicted_{ion_type}"] = (
                    features[f"count_predicted_{ion_type}"] / mask_total
                )
                features[f"fraction_matched_{ion_type}"] = features[
                    f"count_observed_and_predicted_{ion_type}"
                ] / np.maximum(features[f"count_predicted_{ion_type}"], 1.0)

        return features

    def calculate_intensity_statistics(self) -> Dict[str, np.ndarray]:
        """
        计算强度统计特征

        Returns:
            Dict: 强度统计特征字典
        """
        features = {}

        # 计算差异
        diff = self.obs_matrix - self.pred_matrix
        abs_diff = np.abs(diff)

        # 基础统计
        features["mse"] = np.mean(diff**2, axis=1)
        features["mean_abs_diff"] = np.mean(abs_diff, axis=1)
        features["std_abs_diff"] = np.std(abs_diff, axis=1)
        features["max_abs_diff"] = np.max(abs_diff, axis=1)
        features["min_abs_diff"] = np.min(abs_diff, axis=1)

        # 分位数统计
        features["abs_diff_Q1"] = np.percentile(abs_diff, 25, axis=1)
        features["abs_diff_Q2"] = np.percentile(abs_diff, 50, axis=1)  # 中位数
        features["abs_diff_Q3"] = np.percentile(abs_diff, 75, axis=1)

        # 按离子类型统计
        ion_masks = self._create_ion_masks_116()

        for mask_name, mask in ion_masks.items():
            if mask_name in ["y_ion_mask", "b_ion_mask"]:
                ion_type = mask_name.split("_")[0]

                diff_masked = diff * mask[np.newaxis, :]
                abs_diff_masked = np.abs(diff_masked)

                # 只在有效位置计算统计量
                valid_positions = mask > 0

                features[f"mse_{ion_type}"] = np.mean(
                    diff_masked[:, valid_positions] ** 2, axis=1
                )
                features[f"mean_abs_diff_{ion_type}"] = np.mean(
                    abs_diff_masked[:, valid_positions], axis=1
                )
                features[f"std_abs_diff_{ion_type}"] = np.std(
                    diff_masked[:, valid_positions], axis=1
                )

        return features

    def calculate_sequence_features(self) -> Dict[str, np.ndarray]:
        """
        计算序列相关特征

        Returns:
            Dict: 序列特征字典
        """
        features = {}

        # 序列长度
        if "SEQUENCE" in self.metadata.columns:
            features["sequence_length"] = (
                self.metadata["SEQUENCE"].apply(len).values.astype(float)
            )
        else:
            features["sequence_length"] = np.ones(self.n_psm) * 10.0  # 默认长度

        # 电荷特征 (独热编码)
        if "PRECURSOR_CHARGE" in self.metadata.columns:
            charges = self.metadata["PRECURSOR_CHARGE"].values
        else:
            # 默认电荷2：构造长度为 n_psm 的数组
            charges = np.full(self.n_psm, 2)
        for charge in range(1, 7):
            features[f"charge_{charge}"] = (charges == charge).astype(float)

        # 碰撞能量 (如果有的话)
        if "COLLISION_ENERGY" in self.metadata.columns:
            features["collision_energy"] = (
                self.metadata["COLLISION_ENERGY"].values / 100.0
            )
        else:
            features["collision_energy"] = np.ones(self.n_psm) * 0.3  # 默认30% NCE

        # 分子量 (如果有的话)
        if "CALCULATED_MASS" in self.metadata.columns:
            features["mass"] = self.metadata["CALCULATED_MASS"].values
        else:
            features["mass"] = features["sequence_length"] * 110.0  # 平均氨基酸分子量

        return features

    def _calculate_spectral_angle(
        self, obs: np.ndarray, pred: np.ndarray
    ) -> np.ndarray:
        """计算光谱角"""
        return _spectral_angle_impl(obs, pred)

    def _calculate_pearson_correlation(
        self, obs: np.ndarray, pred: np.ndarray
    ) -> np.ndarray:
        """计算皮尔逊相关系数"""
        return _pearson_correlation_impl(obs, pred)

    def _calculate_spearman_correlation(
        self, obs: np.ndarray, pred: np.ndarray
    ) -> np.ndarray:
        """计算斯皮尔曼相关系数"""
        return _spearman_correlation_impl(obs, pred)

    def _calculate_cosine_similarity(
        self, obs: np.ndarray, pred: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度"""
        return _cosine_similarity_impl(obs, pred)

    def calculate_all_features(self) -> pd.DataFrame:
        """
        计算所有特征

        Returns:
            pd.DataFrame: 包含所有特征的DataFrame
        """
        self.logger.info("计算相似性特征...")
        similarity_features = self.calculate_basic_similarity_features()

        self.logger.info("计算碎片统计特征...")
        fragments_features = self.calculate_fragments_statistics()

        self.logger.info("计算强度统计特征...")
        intensity_features = self.calculate_intensity_statistics()

        self.logger.info("计算序列特征...")
        sequence_features = self.calculate_sequence_features()

        # 合并所有特征
        all_features = {
            **similarity_features,
            **fragments_features,
            **intensity_features,
            **sequence_features,
        }

        # 创建DataFrame
        feature_df = pd.DataFrame(all_features)

        # 添加元数据信息
        if "REVERSE" in self.metadata.columns:
            feature_df["Label"] = (
                self.metadata["REVERSE"].apply(lambda x: -1 if x else 1).values
            )
        else:
            feature_df["Label"] = np.ones(self.n_psm)  # 默认都是target

        # 添加标识列：优先使用 metadata 中的 SpecId；否则回退为 PSM_x
        if "SpecId" in self.metadata.columns:
            if len(self.metadata) == self.n_psm:
                try:
                    feature_df["SpecId"] = self.metadata["SpecId"].astype(str).values
                except Exception:
                    # 强制回退到占位 ID
                    self.logger.warning("metadata['SpecId'] 转换失败，使用占位 PSM_x")
                    feature_df["SpecId"] = [f"PSM_{i}" for i in range(self.n_psm)]
            else:
                self.logger.warning(
                    f"metadata 行数({len(self.metadata)})与特征行数({self.n_psm})不一致，使用占位 PSM_x"
                )
                feature_df["SpecId"] = [f"PSM_{i}" for i in range(self.n_psm)]
        else:
            feature_df["SpecId"] = [f"PSM_{i}" for i in range(self.n_psm)]
        feature_df["ScanNr"] = range(self.n_psm)
        feature_df["filename"] = "input_data"

        # 添加肽段信息
        if "SEQUENCE" in self.metadata.columns:
            feature_df["Peptide"] = (
                self.metadata["SEQUENCE"].apply(lambda x: f"_.{x}._").values
            )
            feature_df["Proteins"] = self.metadata["SEQUENCE"].values
        else:
            feature_df["Peptide"] = [f"_.PEPTIDE{i}._" for i in range(self.n_psm)]
            feature_df["Proteins"] = [f"PROTEIN{i}" for i in range(self.n_psm)]

        # 重排列列顺序 (Percolator 格式)
        first_columns = ["SpecId", "Label", "ScanNr", "filename"]
        last_columns = ["Peptide", "Proteins"]
        feature_columns = [
            col for col in feature_df.columns if col not in first_columns + last_columns
        ]

        column_order = first_columns + sorted(feature_columns) + last_columns
        feature_df = feature_df[column_order]

        self.logger.info(f"计算完成，共生成 {len(feature_columns)} 个特征")

        return feature_df


def load_data(
    obs_path: Union[str, Path],
    pred_path: Union[str, Path],
    metadata_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    加载输入数据

    Args:
        obs_path: 观测强度矩阵路径
        pred_path: 预测强度矩阵路径
        metadata_path: 元数据文件路径

    Returns:
        Tuple: (观测矩阵, 预测矩阵, 元数据DataFrame)
    """
    # 加载矩阵
    if str(obs_path).endswith(".npy"):
        obs_matrix = np.load(obs_path)
    elif str(obs_path).endswith(".npz"):
        obs_matrix = np.load(obs_path)["arr_0"]
    else:
        # 尝试加载为文本文件
        obs_matrix = np.loadtxt(obs_path)

    if str(pred_path).endswith(".npy"):
        pred_matrix = np.load(pred_path)
    elif str(pred_path).endswith(".npz"):
        pred_matrix = np.load(pred_path)["arr_0"]
    else:
        pred_matrix = np.loadtxt(pred_path)

    # 加载元数据
    if str(metadata_path).endswith(".csv"):
        metadata = pd.read_csv(metadata_path)
    elif str(metadata_path).endswith(".tsv"):
        metadata = pd.read_csv(metadata_path, sep="\t")
    else:
        # 默认尝试逗号分隔
        metadata = pd.read_csv(metadata_path)

    return obs_matrix, pred_matrix, metadata


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="自定义谱图特征计算工具 - 支持 (N_PSM, 116) 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --obs_matrix observed.npy --pred_matrix predicted.npy --metadata metadata.csv --output features.tab
  %(prog)s -o obs.npy -p pred.npy -m meta.csv --output result.tab --verbose
  %(prog)s --obs_matrix obs.txt --pred_matrix pred.txt --metadata meta.tsv --output features.tab

支持的文件格式:
  矩阵文件: .npy, .npz, .txt (纯文本)
  元数据文件: .csv, .tsv
        """,
    )

    parser.add_argument(
        "--obs_matrix",
        "-o",
        type=str,
        required=True,
        help="观测强度矩阵文件路径 (N_PSM, 116) (必需)",
    )

    parser.add_argument(
        "--pred_matrix",
        "-p",
        type=str,
        required=True,
        help="预测强度矩阵文件路径 (N_PSM, 116) (必需)",
    )

    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        required=True,
        help="元数据文件路径 (CSV/TSV格式) (必需)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="输出特征文件路径 (必需)"
    )

    parser.add_argument(
        "--fragment_dim", type=int, default=116, help="碎片维度 (默认: 116)"
    )

    parser.add_argument(
        "--intensity_threshold",
        type=float,
        default=0.01,
        help="强度阈值，用于判断离子存在 (默认: 0.01)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细输出")

    parser.add_argument(
        "--dry_run", action="store_true", help="试运行，只验证文件格式不计算特征"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("自定义谱图特征计算工具 - 支持 (N_PSM, 116) 格式")
    print("=" * 70)

    # 显示配置
    print(f"观测矩阵: {args.obs_matrix}")
    print(f"预测矩阵: {args.pred_matrix}")
    print(f"元数据文件: {args.metadata}")
    print(f"输出文件: {args.output}")
    print(f"碎片维度: {args.fragment_dim}")
    print(f"强度阈值: {args.intensity_threshold}")

    # 检查输入文件
    for file_path in [args.obs_matrix, args.pred_matrix, args.metadata]:
        if not os.path.isfile(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            sys.exit(1)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        # 加载数据
        print("\n加载输入数据...")
        obs_matrix, pred_matrix, metadata = load_data(
            args.obs_matrix, args.pred_matrix, args.metadata
        )

        print(f"观测矩阵形状: {obs_matrix.shape}")
        print(f"预测矩阵形状: {pred_matrix.shape}")
        print(f"元数据行数: {len(metadata)}")

        if args.verbose:
            print(f"元数据列: {list(metadata.columns)}")

        if args.dry_run:
            print("✓ 试运行模式 - 文件格式验证通过")
            sys.exit(0)

        # 创建特征计算器
        print("\n初始化特征计算器...")
        calculator = CustomPercolatorFeatureCalculator(
            obs_matrix=obs_matrix,
            pred_matrix=pred_matrix,
            metadata=metadata,
            fragment_dim=args.fragment_dim,
        )

        # 计算特征
        print("\n开始特征计算...")
        feature_df = calculator.calculate_all_features()

        # 保存结果
        print(f"\n保存结果到: {args.output}")
        feature_df.to_csv(args.output, sep="\t", index=False)

        # 显示统计信息
        print("\n" + "=" * 70)
        print("✓ 特征计算完成！")
        print(f"输入PSM数量: {len(obs_matrix)}")
        print(f"生成特征数量: {len(feature_df.columns) - 6}")  # 排除元数据列
        print(f"输出文件: {args.output}")

        if os.path.exists(args.output):
            file_size_mb = os.path.getsize(args.output) / 1024 / 1024
            print(f"文件大小: {file_size_mb:.2f} MB")

        # 显示主要特征的统计
        if args.verbose:
            print("\n主要特征统计:")
            key_features = [
                "spectral_angle",
                "pearson_corr",
                "count_observed_and_predicted",
                "mse",
            ]
            for feature in key_features:
                if feature in feature_df.columns:
                    values = feature_df[feature].values
                    print(
                        f"  {feature}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}"
                    )

        print("=" * 70)

    except Exception as e:
        logger.error(f"特征计算失败: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
