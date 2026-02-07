#!/usr/bin/env python3
"""
DeepFLR3.3: Compute spectral similarity score between predicted and reference spectra
===================================================================================

- 依据 7.1computer_mamba_loss.py 的谱图距离计算逻辑
- 使用 Raw_file + MS2_Scan_Number 在 pred_h5 与 ref_h5 间对齐
- 支持多 n（在 [:, :, 0:n] 上计算）与两种模式（flatten / per-position）
- 将 loss 写回 pred_h5
- 同时把 score 写入指定的模板 CSV（默认: script/test_modelresult_template.csv 的 score 列）

Usage examples:
    # 单一 n，默认 flatten
    python script/DeepFLR3.3_mamba_loss.py \
        --pred_h5 script/test_mamba_input.h5 \
        --ref_h5 /mnt/data_nas/lcy/project_MS2predict/1.data/independent_test/PTM/PXD000138/rescore/rescore_batch1.h5 \
        --pred_key Intpredict --true_key train_data \
        --n 31 \
        --template_csv script/test_modelresult_template.csv

    # 多 n，并在 per-position 模式
    python script/DeepFLR3.3_mamba_loss.py \
        --pred_h5 script/test_mamba_input.h5 \
        --ref_h5 /.../rescore_batch1.h5 \
        --n 5 6 7 8 \
        --mode per-position \
        --template_csv script/test_modelresult_template.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import sys

# 导入 spectrum_utils 用于计算理论碎片 m/z
try:
    from spectrum_utils import fragment_annotation, proforma
except ImportError:
    fragment_annotation = None
    proforma = None


def _to_text(value) -> str:
    """将 h5py/np 标量安全转换为 str（避免 bytes 被 str() 变成 \"b'...'\"）。"""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


# 只按网格内离子做冲突检测：预先加载离子注释矩阵，避免在 2w+ 样本循环中重复导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
EXTRACT_ROOT = os.path.join(REPO_ROOT, "spectrum_processing", "extract_real_spectrums")
if EXTRACT_ROOT not in sys.path:
    sys.path.insert(0, EXTRACT_ROOT)

try:
    from step3_generate_train_data import ANNOTATION_MATRIX  # type: ignore
except Exception:
    ANNOTATION_MATRIX = None  # type: ignore


def compute_theoretical_mz_grid(annotate: str, charge: int) -> np.ndarray:
    """
    计算单条肽段序列在 29×31 网格中每个位置的理论 m/z 值。
    
    参数:
        annotate: ProForma 格式序列（如 "ALLS[Phospho]LATHK"）
        charge: 前体电荷
    
    返回:
        mz_grid: (29, 31) 的 m/z 矩阵，无对应离子的位置为 NaN
    """
    if proforma is None or fragment_annotation is None:
        raise ImportError("需要 spectrum_utils 库来计算理论 m/z")
    if ANNOTATION_MATRIX is None:
        raise ImportError("无法导入 step3_generate_train_data.ANNOTATION_MATRIX，无法映射离子网格")
    
    # 解析序列
    try:
        seq = proforma.parse(annotate)
        if not seq:
            return np.full((29, 31), np.nan, dtype=np.float32)
        seq = seq[0]
    except Exception:
        return np.full((29, 31), np.nan, dtype=np.float32)
    
    # 生成理论碎片（包括 b/y/m/immonium，以及中性丢失）
    try:
        theoretical_fragments = fragment_annotation.get_theoretical_fragments(
            seq,
            ion_types="byIm",
            max_charge=2,
            neutral_losses={"H3PO4": -97.976896},
        )
    except Exception:
        return np.full((29, 31), np.nan, dtype=np.float32)
    
    # 构建离子名称到 m/z 的映射
    ion_mz_map = {}
    for fragment, mz_value in theoretical_fragments:
        name = str(fragment)
        if mz_value is not None:
            ion_mz_map[name] = float(mz_value)
    
    # 填充 m/z 网格
    mz_grid = np.full((29, 31), np.nan, dtype=np.float32)
    for row in range(29):
        for col in range(31):
            ion_name = ANNOTATION_MATRIX[col, row]  # 注意：ANNOTATION_MATRIX 是 (31, 29)
            if ion_name and ion_name in ion_mz_map:
                mz_grid[row, col] = ion_mz_map[ion_name]
    
    return mz_grid


def generate_by_priority_mask(mz_grid: np.ndarray, mass_analyzer: str = "FTMS") -> np.ndarray:
    """
    根据理论 m/z 网格生成 b/y 优先 mask。
    
    约定：
    - mz_grid 形状为 (29, V)，与 H5 光谱张量一致：第 0 维为位置（length），第 1 维为离子通道（channel）。
    - by 集合：仅包含 b*/y* 离子（不包含 immonium：IH/IR/IF/IY）。
    - m 系列：仅对 m* 位置做冲突检测；若 m 的理论 m/z 与任一 b/y **完全相等**，则置 0。
      - 不考虑 ppm/Da 容差；只要 m/z 值一致就认为冲突，不一致则不冲突。
    
    参数:
        mz_grid: (29, V) 的理论 m/z 矩阵
        mass_analyzer: 兼容保留参数（当前 exact m/z 判定不使用）
    
    返回:
        mask: (29, V) 的二值 mask，1 表示保留，0 表示置零
    """
    if ANNOTATION_MATRIX is None:
        raise ImportError("无法导入 step3_generate_train_data.ANNOTATION_MATRIX，无法生成 b/y 优先 mask")

    L, V = mz_grid.shape
    # 默认全保留
    mask = np.ones((L, V), dtype=np.uint8)

    # 收集所有 b/y 理论 m/z（按离子名称判定，自动排除 immonium）
    by_mz_set: set[float] = set()
    for row in range(L):
        for col in range(V):
            ion_name = ANNOTATION_MATRIX[col, row]
            if (
                isinstance(ion_name, str)
                and len(ion_name) >= 2
                and ion_name[0] in ("b", "y")
                and ion_name[1].isdigit()
            ):
                mz = mz_grid[row, col]
                if not np.isnan(mz):
                    by_mz_set.add(float(mz))

    if not by_mz_set:
        return mask

    # 仅对 m* 位置做冲突检测
    for row in range(L):
        for col in range(V):
            ion_name = ANNOTATION_MATRIX[col, row]
            if not (isinstance(ion_name, str) and ion_name.startswith("m")):
                continue

            m_mz = mz_grid[row, col]
            if np.isnan(m_mz):
                continue

            if float(m_mz) in by_mz_set:
                mask[row, col] = 0

    return mask


def apply_by_priority_to_batch(
    spectra: np.ndarray,
    sequences: np.ndarray,
    charges: np.ndarray,
    mass_analyzers: np.ndarray,
    num_workers: int = 1,
    copy_input: bool = True,
) -> np.ndarray:
    """
    对一批预测谱图应用 b/y 优先逻辑（基于精确 m/z 冲突检测）。
    
    参数:
        spectra: (N, 29, 31) 预测谱图强度矩阵
        sequences: (N,) ProForma 格式序列数组
        charges: (N,) 电荷数组
        mass_analyzers: (N,) 质量分析器类型数组
        num_workers: 线程数（<=1 表示单线程）
        copy_input: 是否复制输入谱图后再处理。若为 False，则会就地修改 spectra（更快、占用更少内存）。
    
    返回:
        处理后的谱图矩阵（冲突的 m 离子位置置零）
    """
    if spectra.ndim != 3:
        raise ValueError(f"输入谱图维度应为 3，实际为 {spectra.ndim}")
    
    N, L, V = spectra.shape
    if len(sequences) != N or len(charges) != N or len(mass_analyzers) != N:
        raise ValueError(
            f"批量维度不一致: spectra N={N}, sequences={len(sequences)}, charges={len(charges)}, "
            f"mass_analyzers={len(mass_analyzers)}"
        )
    result = spectra.copy() if copy_input else spectra

    # 只按网格内离子 + 精确相等判定：mask 只依赖序列与 charge（当前理论碎片生成 max_charge=2，不依赖 analyzer）
    # 因此按 (Sequence, Charge) 分组复用 mask，最大化缓存命中
    key_to_indices: dict[tuple[str, int], list[int]] = {}
    for i in range(N):
        seq = _to_text(sequences[i])
        charge = int(charges[i])
        key = (seq, charge)
        key_to_indices.setdefault(key, []).append(i)

    unique_keys = list(key_to_indices.keys())
    masks: dict[tuple[str, int], np.ndarray] = {}

    def _compute_mask_for_key(key: tuple[str, int]) -> tuple[tuple[str, int], np.ndarray]:
        seq, charge = key
        mz_grid = compute_theoretical_mz_grid(seq, charge)
        if V > mz_grid.shape[1]:
            raise ValueError(f"谱图通道数 V={V} 超过理论网格通道数 {mz_grid.shape[1]}")
        mz_grid = mz_grid[:, :V]
        mask = generate_by_priority_mask(mz_grid)
        return key, mask

    # 多线程计算每个 unique key 的 mask
    if num_workers > 1 and len(unique_keys) > 1:
        max_workers = min(int(num_workers), len(unique_keys))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for key, mask in ex.map(_compute_mask_for_key, unique_keys):
                masks[key] = mask
    else:
        for key in unique_keys:
            k, mask = _compute_mask_for_key(key)
            masks[k] = mask

    # 按组批量应用 mask（广播乘法）
    for key, idxs in key_to_indices.items():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        result[idx_arr] *= masks[key]

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute spectral similarity between pred_h5[pred_key] and ref_h5[true_key], "
            "aligned by (Raw_file, MS2_Scan_Number). Write loss to pred_h5 and score to template CSV."
        )
    )
    parser.add_argument("--pred_h5", required=True, help="预测 H5 文件 (包含 Intpredict)。")
    parser.add_argument("--ref_h5", required=True, help="参考 H5 文件 (包含 train_data)。")
    parser.add_argument("--pred_key", default="Intpredict", help="预测数据集名称，默认 Intpredict。")
    parser.add_argument("--true_key", default="train_data", help="真实数据集名称，默认 train_data。")
    parser.add_argument(
        "--n", required=True, nargs="+", help="截取预测/真实张量的前 n 个通道，可写多个值或列表（如 5 6 7 或 [5,6,7]）。"
    )
    parser.add_argument(
        "--mode",
        choices=["flatten", "per-position"],
        default="flatten",
        help="谱距计算方式 (默认: flatten)。",
    )
    parser.add_argument(
        "--align",
        choices=["scan", "index"],
        default="scan",
        help=(
            "预测与参考的对齐方式：scan=按 (Raw_file, MS2_Scan_Number) 对齐 "
            "（兼容旧流程），index=按行号一一对应（新 target_decoy 候选级流程）。"
        ),
    )
    parser.add_argument(
        "--template_csv",
        default="script/test_modelresult_template.csv",
        help="要写入 score 列的模板 CSV 路径 (默认: script/test_modelresult_template.csv)",
    )
    parser.add_argument(
        "--by_priority_workers",
        type=int,
        default=max(1, min(120, os.cpu_count() or 1)),
        help="b/y 优先预处理线程数（<=1 表示单线程）。默认=min(120, CPU 核数)。",
    )
    return parser.parse_args()


def _split_n_tokens(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    token = token.replace(",", " ")
    return [piece for piece in token.split() if piece]


def parse_n_values(raw_values: List[str]) -> List[int]:
    if not raw_values:
        raise ValueError("--n 必须指定至少一个数值。")

    tokens: List[str] = []
    if len(raw_values) == 1:
        token = raw_values[0].strip()
        if token.startswith("[") and token.endswith("]"):
            token = token[1:-1]
        tokens.extend(_split_n_tokens(token))
    else:
        for value in raw_values:
            tokens.extend(_split_n_tokens(value))

    if not tokens:
        raise ValueError("--n 解析后为空，请提供正整数列表。")

    result: List[int] = []
    seen = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"--n 仅支持正整数，收到: {value}")
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def as_str_array(ds) -> np.ndarray:
    """Decode an HDF5 bytes dataset to numpy array of str."""
    try:
        return ds.asstr()[:]
    except AttributeError:
        # Older h5py might not have asstr; fallback
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in ds[:]])


def load_aligned_pairs(
    pred_h5: str, ref_h5: str, pred_key: str, true_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pred/ref arrays and align indices via (Raw_file, MS2_Scan_Number).

    Returns:
        y_pred_full: (N_pred, L, V) float32
        y_true_full: (N_ref, L, V) float32
        pred_raw: (N_pred,) str
        pred_scan: (N_pred,) int
        ref_raw: (N_ref,) str
        ref_scan: (N_ref,) int
        pred_sequences: (N_pred,) str - ProForma 格式序列
        pred_charges: (N_pred,) int - 电荷
        pred_analyzers: (N_pred,) str - 质量分析器类型
        by_priority_mask: (N_ref, L, V) uint8 - 预计算的 b/y 优先 mask（若不存在则为 None）
    """
    with h5py.File(pred_h5, "r") as fp:
        if pred_key not in fp:
            raise KeyError(f"在 {pred_h5} 中未找到预测数据集: {pred_key}")
        if "Raw_file" not in fp or "MS2_Scan_Number" not in fp:
            raise KeyError("预测 H5 缺少 Raw_file 或 MS2_Scan_Number 数据集，请用更新后的 DeepFLR3.1 生成。")
        y_pred_full = fp[pred_key][:].astype(np.float32, copy=False)
        pred_raw = as_str_array(fp["Raw_file"])
        pred_scan = fp["MS2_Scan_Number"][:].astype(np.int64, copy=False)
        
        # 加载序列、电荷和质量分析器信息（用于 b/y 优先处理的后备方案）
        pred_sequences = as_str_array(fp["Sequence"]) if "Sequence" in fp else None
        pred_charges = fp["Charge"][:].astype(np.int64, copy=False) if "Charge" in fp else None
        pred_analyzers = as_str_array(fp["Mass_analyzer"]) if "Mass_analyzer" in fp else None

    with h5py.File(ref_h5, "r") as fr:
        if true_key not in fr:
            raise KeyError(f"在 {ref_h5} 中未找到真实数据集: {true_key}")
        if "Raw_file" not in fr or "MS2_Scan_Number" not in fr:
            raise KeyError("参考 H5 缺少 Raw_file 或 MS2_Scan_Number 数据集。")
        y_true_full = fr[true_key][:].astype(np.float32, copy=False)
        ref_raw = as_str_array(fr["Raw_file"])
        ref_scan = fr["MS2_Scan_Number"][:].astype(np.int64, copy=False)
        
        # 读取预计算的 b/y 优先 mask（若存在）
        by_priority_mask = fr["by_priority_mask"][:].astype(np.uint8, copy=False) if "by_priority_mask" in fr else None

    return y_pred_full, y_true_full, pred_raw, pred_scan, ref_raw, ref_scan, pred_sequences, pred_charges, pred_analyzers, by_priority_mask


def build_alignment(
    pred_raw: np.ndarray,
    pred_scan: np.ndarray,
    ref_raw: np.ndarray,
    ref_scan: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build aligned index pairs via (Raw_file, MS2_Scan_Number)."""
    # Build ref map
    ref_map = {}
    for j, (r, s) in enumerate(zip(ref_raw, ref_scan)):
        ref_map[(str(r), int(s))] = j

    pred_to_ref = []
    matched_pred_idx = []
    missing = 0
    for i, (r, s) in enumerate(zip(pred_raw, pred_scan)):
        key = (str(r), int(s))
        j = ref_map.get(key)
        if j is None:
            missing += 1
            continue
        matched_pred_idx.append(i)
        pred_to_ref.append(j)

    return np.array(matched_pred_idx, dtype=np.int64), np.array(pred_to_ref, dtype=np.int64), missing


def masked_spectral_distance_flatten(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """将 (L,V) 展平为一维向量，按 y_true != -1 做掩码后应用 Sqrt+L2 变换计算余弦相似度，结果范围在 [-1,1]。"""
    b, l, v = y_true.shape
    y_true_flat = y_true.reshape(b, -1)
    y_pred_flat = y_pred.reshape(b, -1)
    mask = (y_true_flat != -1).float()
    y_true_masked = y_true_flat * mask
    y_pred_masked = y_pred_flat * mask
    # 应用 Sqrt 变换
    y_true_sqrt = torch.sqrt(y_true_masked)
    y_pred_sqrt = torch.sqrt(y_pred_masked)
    # L2 归一化
    y_true_norm = F.normalize(y_true_sqrt, p=2, dim=-1)
    y_pred_norm = F.normalize(y_pred_sqrt, p=2, dim=-1)
    # 直接使用归一化向量的点积作为余弦相似度
    cos_sim = torch.sum(y_true_norm * y_pred_norm, dim=-1).clamp(-1.0, 1.0)
    return cos_sim  # (B,)


def masked_spectral_distance_perpos(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """逐位置应用 Sqrt+L2 变换计算余弦相似度，然后在长度维上取平均，结果范围在 [-1,1]。"""
    mask = (y_true != -1).float()
    y_true_m = y_true * mask
    y_pred_m = y_pred * mask
    # 应用 Sqrt 变换
    y_true_sqrt = torch.sqrt(y_true_m)
    y_pred_sqrt = torch.sqrt(y_pred_m)
    # L2 归一化
    y_true_norm = F.normalize(y_true_sqrt, p=2, dim=-1)
    y_pred_norm = F.normalize(y_pred_sqrt, p=2, dim=-1)
    # 每个 (B,L) 位置的归一化向量点积即为余弦相似度
    cos_sim = torch.sum(y_true_norm * y_pred_norm, dim=-1).clamp(-1.0, 1.0)  # (B,L)
    return cos_sim.mean(dim=-1)  # (B,)


def compute_scores(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    mode: str,
    extra_mask_np: np.ndarray | None = None,
) -> np.ndarray:
    """
    根据给定的真实/预测谱图计算谱图相似度。

    若 extra_mask_np 不为 None，则会先将 mask==0 的位置在 y_true 中标记为 -1，
    使得 masked_spectral_distance_* 仅在 mask==1 的位置上工作。
    """
    y_true = torch.tensor(y_true_np, dtype=torch.float32)
    y_pred = torch.tensor(y_pred_np, dtype=torch.float32)

    if extra_mask_np is not None:
        if extra_mask_np.shape != y_true_np.shape:
            raise ValueError(
                f"extra_mask_np 形状 {extra_mask_np.shape} 与 y_true 形状 {y_true_np.shape} 不一致"
            )
        mask_bool = torch.tensor(extra_mask_np != 0, dtype=torch.bool)
        # 将非修饰相关位置标记为 -1，复用已有的 masked 逻辑
        y_true = y_true.masked_fill(~mask_bool, -1.0)

    if mode == "flatten":
        scores = masked_spectral_distance_flatten(y_true, y_pred)
    else:
        scores = masked_spectral_distance_perpos(y_true, y_pred)
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def write_dataset(pred_h5: str, name: str, data: np.ndarray) -> None:
    with h5py.File(pred_h5, "a") as f:
        if name in f:
            del f[name]
        f.create_dataset(name, data=data)
        print(f"写入数据集: {name}")


def update_template_csv(template_csv: str, scores_full: np.ndarray) -> None:
    df = pd.read_csv(template_csv)
    if len(df) != len(scores_full):
        raise ValueError(
            f"模板 CSV 行数 {len(df)} 与预测样本数 {len(scores_full)} 不一致，无法按索引对齐写入 score。"
        )
    df["score"] = scores_full
    df.to_csv(template_csv, index=False)


def main() -> None:
    args = parse_args()
    n_values = parse_n_values(args.n)

    # 1) 加载预测与参考张量（包括序列、电荷、分析器信息和预计算的 mask）
    (
        y_pred_full,
        y_true_full,
        pred_raw,
        pred_scan,
        ref_raw,
        ref_scan,
        pred_sequences,
        pred_charges,
        pred_analyzers,
        by_priority_mask_full,
    ) = load_aligned_pairs(args.pred_h5, args.ref_h5, args.pred_key, args.true_key)

    if y_pred_full.shape[1] != y_true_full.shape[1]:
        raise ValueError(
            f"长度维不一致: pred L={y_pred_full.shape[1]} vs true L={y_true_full.shape[1]}"
        )

    # 2) 构建对齐索引
    if args.align == "scan":
        matched_pred_idx, matched_ref_idx, missing = build_alignment(
            pred_raw, pred_scan, ref_raw, ref_scan
        )
        N_pred = y_pred_full.shape[0]
    else:
        # index 模式：要求预测与参考样本数一致，按行号一一对齐
        N_pred = y_pred_full.shape[0]
        N_ref = y_true_full.shape[0]
        if N_pred != N_ref:
            raise ValueError(
                f"index 对齐模式下样本数不一致: pred N={N_pred} vs true N={N_ref}"
            )
        matched_pred_idx = np.arange(N_pred, dtype=np.int64)
        matched_ref_idx = np.arange(N_ref, dtype=np.int64)
        missing = 0

    # 将未匹配的样本最终标记为 NaN

    multi = len(n_values) > 1

    # 2) 多 n 计算
    max_pred_channels = y_pred_full.shape[2]
    max_true_channels = y_true_full.shape[2]

    # 仅需对齐一次：预先裁剪到最大 n，并（可选）只做一次 b/y 优先预处理
    max_n = max(n_values)
    if max_n > max_pred_channels or max_n > max_true_channels:
        raise ValueError(
            f"max(n)={max_n} 超过张量第三维: pred={max_pred_channels}, true={max_true_channels}"
        )

    # 预测谱图：对齐 + 只裁剪到 max_n（后续各 n 直接切片视图，避免重复拷贝）
    y_pred_matched_max = y_pred_full[:, :, :max_n][matched_pred_idx]

    # 只在多 n 时，才预先对齐/裁剪真实谱图（避免单 n 占用额外内存）
    if multi:
        y_true_matched_max = y_true_full[:, :, :max_n][matched_ref_idx]
    else:
        y_true_matched_max = None

    # 释放原始预测张量以降低峰值内存占用（后续只使用 y_pred_matched_max）
    del y_pred_full

    # 对预测谱图和真实谱图应用 b/y 优先逻辑（对称处理）
    # 优先使用预计算的 mask（来自 step3），若不存在则动态计算
    if by_priority_mask_full is not None:
        # 获取对齐后的 mask（与 matched_ref_idx 对应）
        mask_matched = by_priority_mask_full[matched_ref_idx][:, :, :max_n]
        # 对预测谱图应用 mask
        y_pred_matched_max *= mask_matched.astype(np.float32)
        # 对真实谱图也应用相同的 mask（对称处理）
        if multi and y_true_matched_max is not None:
            y_true_matched_max *= mask_matched.astype(np.float32)
    elif pred_sequences is not None and pred_charges is not None and pred_analyzers is not None:
        # 后备方案：动态计算 mask（仅处理预测谱图，与旧逻辑一致）
        seq_crop = pred_sequences[matched_pred_idx]
        charge_crop = pred_charges[matched_pred_idx]
        analyzer_crop = pred_analyzers[matched_pred_idx]

        apply_by_priority_to_batch(
            y_pred_matched_max,
            seq_crop,
            charge_crop,
            analyzer_crop,
            num_workers=args.by_priority_workers,
            copy_input=False,
        )
    else:
        pass  # 缺少预计算 mask 和序列/电荷/分析器信息，跳过 b/y 优先处理

    for idx, n in enumerate(n_values, start=1):
        if n > max_pred_channels or n > max_true_channels:
            raise ValueError(
                f"n={n} 超过张量第三维: pred={max_pred_channels}, true={max_true_channels}"
            )

        # 截取到当前 n（y_pred 直接从已对齐/已预处理的 max_n 视图切片）
        y_pred_crop = y_pred_matched_max[:, :, :n]
        if multi:
            assert y_true_matched_max is not None
            y_true_crop = y_true_matched_max[:, :, :n]
        else:
            y_true_crop = y_true_full[:, :, :n][matched_ref_idx].copy()
            # 单 n 情况下，也对真实谱图应用预计算的 mask（对称处理）
            if by_priority_mask_full is not None:
                mask_crop = by_priority_mask_full[matched_ref_idx][:, :, :n]
                y_true_crop *= mask_crop.astype(np.float32)

        scores_part = compute_scores(
            y_true_crop,
            y_pred_crop,
            args.mode,
            extra_mask_np=None,
        )  # 长度 = 匹配数

        # 组装为全量 N_pred 长度，未匹配填 NaN
        scores_full = np.full((N_pred,), np.nan, dtype=np.float32)
        scores_full[matched_pred_idx] = scores_part

        # 3) 写回 H5
        ds_name = "Cosine_Similarity"
        write_dataset(args.pred_h5, ds_name, scores_full)

        # 4) 写入模板 CSV（覆盖/新增 score 列）。若多 n，可在外部重复运行或修改为写 score_n{n}
        try:
            update_template_csv(args.template_csv, scores_full)
        except Exception:
            pass  # 静默处理 CSV 更新失败


if __name__ == "__main__":
    main()
