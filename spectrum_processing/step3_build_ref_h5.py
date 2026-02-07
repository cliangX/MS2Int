#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFLR3.1b: 从 DeepFLR target_decoy 序列生成候选级参考谱图 H5
===============================================================

目标：
- 输入：DeepFLR1_step1_target_decoy.csv + 原始 msms.txt + mzML 目录
- 输出：候选级参考谱图 H5（每一行对应一个 target/decoy 候选，包含 train_data）
- 用途：与 Mamba 预测的 Intpredict 一一对应，供 DeepFLR3.3_mamba_loss 计算谱图相似度分数

实现要点：
- 使用 DeepFLR3.1_csv_to_h5 中的 convert_deepflr_to_mamba_sequence 将 key 序列转为 ProForma 风格 annotate
- 复用 extract_real_spectrums.step2/step3 中的碎片生成与强度匹配逻辑（cached_process_single, fast_intensity_matching, ION_TO_IDX 等）
- 按 (SourceFile, Spectrum) 从 mzML 中提取 MS2 峰列表，并与 annotate 对应的理论碎片做匹配
- 将每个候选的碎片强度映射到固定 31×29 网格，再转置为 (29,31) 存入 train_data
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np
import pandas as pd
import pyopenms as oms
import re
from tqdm import tqdm


# ---------------------------------------------------------------------
# 导入项目内已有工具函数
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
# extract_real_spectrums 目录位于当前脚本同级目录下
EXTRACT_ROOT = os.path.join(CURRENT_DIR, "extract_real_spectrums")

if EXTRACT_ROOT not in sys.path:
    sys.path.insert(0, EXTRACT_ROOT)

# 复用碎片生成与匹配的核心逻辑
from step2_process_df_h5 import (  # type: ignore
    cached_process_single,
    fast_intensity_matching,
)
from step3_generate_train_data import ION_ROWS, ION_COLS, ION_TO_IDX, ANNOTATION_MATRIX  # type: ignore

# 尝试导入 spectrum_utils 用于计算理论碎片 m/z
try:
    from spectrum_utils import proforma, fragment_annotation
    SPECTRUM_UTILS_AVAILABLE = True
except ImportError:
    proforma = None  # type: ignore
    fragment_annotation = None  # type: ignore
    SPECTRUM_UTILS_AVAILABLE = False


def _normalize_ion_name(name: str):
    """
    名称归一化（与 extract_real_spectrums/step3 保持一致）：
    - 去掉磷酸中性丢失后缀 -H3PO4；
    - b/y 等常规离子保留电荷后缀（^2 等）；
    - m 系列忽略电荷，将 m2:4、m2:4^2、m2:4-H3PO4 等统一归一到 m2:4。
    
    这样可以将带中性丢失的碎片强度累加到对应的基础离子上。
    """
    if not name:
        return None

    base = str(name)
    charge_suffix = ""
    
    # 关键：先移除中性丢失标记，再分离电荷。
    # 兼容不同命名风格：
    # - y9-H3PO4^2（中性丢失在电荷前）
    # - b5^2-H3PO4（中性丢失在电荷后）
    if "-H3PO4" in base:
        base = base.replace("-H3PO4", "")
    
    # 然后分离电荷后缀
    if "^" in base:
        parts = base.split("^", 1)
        base, charge_suffix = parts[0], "^" + parts[1]

    # 若 base 为空，直接丢弃
    if not base:
        return None

    # m 系列：忽略电荷，把不同电荷和是否带 -H3PO4 的 m 碎片统一到同一个 key
    # 例如 m2:4, m2:4^2, m2:4-H3PO4, m2:4-H3PO4^2 均归一为 m2:4
    if base.startswith("m"):
        return base

    # 其他离子（b/y/Im 等）保留电荷信息，用于区分 1+ / 2+
    return base + charge_suffix


def compute_theoretical_mz_grid(annotate: str, charge: int) -> np.ndarray:
    """
    计算单条肽段序列在 29×31 网格中每个位置的理论 m/z 值。
    
    参数:
        annotate: ProForma 格式序列（如 "ALLS[Phospho]LATHK"）
        charge: 前体电荷
    
    返回:
        mz_grid: (29, 31) 的 m/z 矩阵，无对应离子的位置为 NaN
    """
    if not SPECTRUM_UTILS_AVAILABLE:
        raise ImportError("需要 spectrum_utils 库来计算理论 m/z")
    if ANNOTATION_MATRIX is None:
        raise ImportError("无法导入 ANNOTATION_MATRIX，无法映射离子网格")
    
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
    # theoretical_fragments 返回 [(ion_name, mz), ...] 格式
    ion_mz_map = {}
    for ion_name, mz_value in theoretical_fragments:
        if mz_value is not None:
            ion_mz_map[str(ion_name)] = float(mz_value)
    
    # 填充 m/z 网格
    mz_grid = np.full((29, 31), np.nan, dtype=np.float32)
    for row in range(29):
        for col in range(31):
            ion_name = ANNOTATION_MATRIX[col, row]  # 注意：ANNOTATION_MATRIX 是 (31, 29)
            if ion_name and ion_name in ion_mz_map:
                mz_grid[row, col] = ion_mz_map[ion_name]
    
    return mz_grid


def generate_by_priority_mask(mz_grid: np.ndarray) -> np.ndarray:
    """
    根据理论 m/z 网格生成 b/y 优先 mask。
    
    逻辑：
    - 收集所有 b*/y* 离子的理论 m/z
    - 对于 m* 离子，若其理论 m/z 与任一 b/y 完全相等，则置 0
    
    参数:
        mz_grid: (29, 31) 的理论 m/z 矩阵
    
    返回:
        mask: (29, 31) 的二值 mask，1 表示保留，0 表示置零
    """
    if ANNOTATION_MATRIX is None:
        raise ImportError("无法导入 ANNOTATION_MATRIX，无法生成 b/y 优先 mask")

    L, V = mz_grid.shape  # (29, 31)
    mask = np.ones((L, V), dtype=np.uint8)

    # 收集所有 b/y 理论 m/z（按离子名称判定，排除 immonium）
    by_mz_set: set = set()
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


def compute_by_priority_mask_for_sequence(annotate: str, charge: int) -> np.ndarray:
    """
    为单条肽段序列计算 b/y 优先 mask。
    
    参数:
        annotate: ProForma 格式序列
        charge: 前体电荷
    
    返回:
        mask: (29, 31) 的二值 mask
    """
    if not SPECTRUM_UTILS_AVAILABLE:
        # 若无 spectrum_utils，返回全 1 mask（不做 mask 处理）
        return np.ones((29, 31), dtype=np.uint8)
    
    mz_grid = compute_theoretical_mz_grid(annotate, charge)
    return generate_by_priority_mask(mz_grid)


def convert_deepflr_to_mamba_sequence(key_x: str) -> str:
    """
    将 DeepFLR 编码序列（含 1/2/3/4）转换为 ProForma 风格序列。
    逻辑与 DeepFLR3.1_csv_to_h5.convert_deepflr_to_mamba_sequence 保持一致。
    """
    modification_map = {
        "1": "[Phospho]",
        "2": "[Oxidation]",
        "3": "[Carbamidomethyl]",
        "4": "[Acetyl]",
    }

    import re

    mamba_sequence = key_x

    # N 端乙酰化：4 开头
    if mamba_sequence.startswith("4"):
        mamba_sequence = "[Acetyl]-" + mamba_sequence[1:]

    # 其他修饰：X1/X2/X3 -> X[Mod]
    for deepflr_code, mamba_mod in modification_map.items():
        if deepflr_code == "4":
            continue
        pattern = f"([A-Z]){deepflr_code}"
        replacement = f"\\1{mamba_mod}"
        mamba_sequence = re.sub(pattern, replacement, mamba_sequence)

    return mamba_sequence


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------

def load_msms_msinfo(
    msms_path: str,
) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    """
    从 msms.txt 中读取：
    - (Raw file, Scan number) -> Mass analyzer
    - (Raw file, Scan number) -> Fragmentation

    Mass analyzer 用于选择匹配窗口的 ppm / Da 容差；
    Fragmentation 仅作为元数据写入 H5，方便后续分析。
    """
    df = pd.read_csv(msms_path, sep="\t", low_memory=False)

    required_cols = ["Raw file", "Scan number", "Mass analyzer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"msms.txt 缺少必要列: {col}")

    has_frag = "Fragmentation" in df.columns

    df["Raw file"] = df["Raw file"].astype(str)
    df["Scan number"] = pd.to_numeric(df["Scan number"], errors="coerce").astype("Int64")

    msinfo: Dict[Tuple[str, int], str] = {}
    fraginfo: Dict[Tuple[str, int], str] = {}
    for _, row in df.iterrows():
        raw = row["Raw file"]
        scan = row["Scan number"]
        analyzer = row["Mass analyzer"]
        if pd.isna(scan) or pd.isna(analyzer):
            continue
        key = (raw, int(scan))
        msinfo[key] = str(analyzer)
        if has_frag:
            frag = row["Fragmentation"]
            if not pd.isna(frag):
                fraginfo[key] = str(frag)

    return msinfo, fraginfo


def load_mzml_ms2_map(
    mzml_path: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    加载单个 mzML，返回：
        MS2_scan_number -> (mz_array, intensity_array, RT, collision_energy)
    这里的 scan number 定义与 extract_real_spectrums.step2 保持一致：使用 exp 中谱图的索引 + 1。
    """
    if not os.path.isfile(mzml_path):
        raise FileNotFoundError(f"找不到 mzML 文件: {mzml_path}")

    exp = oms.MSExperiment()
    oms.MzMLFile().load(mzml_path, exp)

    # 与 extract_real_spectrums 一致的归一化
    normalizer = oms.Normalizer()
    param = normalizer.getParameters()
    param.setValue("method", "to_one")
    normalizer.setParameters(param)
    normalizer.filterPeakMap(exp)

    ms2_map: Dict[int, Tuple[np.ndarray, np.ndarray, float, float]] = {}

    for idx, spectrum in enumerate(exp):
        if spectrum.getMSLevel() != 2:
            continue
        if not spectrum.getPrecursors():
            continue

        mz_array, int_array = spectrum.get_peaks()
        mz = np.asarray(mz_array, dtype=float)
        inten = np.asarray(int_array, dtype=float)
        rt = float(spectrum.getRT())
        ce = float("nan")
        try:
            precursors = spectrum.getPrecursors()
            if precursors:
                precursor = precursors[0]
                if precursor.metaValueExists("collision energy"):
                    ce = float(precursor.getMetaValue("collision energy"))
        except Exception:
            # 若无法解析碰撞能量，则保持为 NaN，后续可根据需要填默认值
            ce = float("nan")

        scan_number = idx + 1  # 与 msms.txt 的 Scan number 一致的约定
        ms2_map[scan_number] = (mz, inten, rt, ce)

    return ms2_map


def build_train_matrix(frag_int_pairs: List[Tuple[str, float]]) -> np.ndarray:
    """
    将 [ (ion_name, intensity), ... ] 映射到固定 31×29 网格，再返回 (31,29) 的矩阵。
    
    关键改进（与 extract_real_spectrums/step3 保持一致）：
    - 使用 _normalize_ion_name 归一化离子名称（去除 -H3PO4 后缀）
    - 使用累加逻辑（vec[idx] += inten），将中性丢失碎片强度累加到基础离子上
    - 例如：b5 和 b5-H3PO4 的强度会累加到同一个 b5 位置
    """
    flat_len = ION_ROWS * ION_COLS
    vec = np.zeros(flat_len, dtype=float)

    for name, inten in frag_int_pairs:
        # 归一化离子名称：去除 -H3PO4 后缀，统一 m 系列电荷等
        norm_name = _normalize_ion_name(name)
        if norm_name is None:
            continue
        
        idx = ION_TO_IDX.get(norm_name)
        if idx is None:
            continue
        
        # 累加强度（而非覆盖），实现中性丢失碎片的合并
        vec[idx] += float(inten)

    return vec.reshape((ION_ROWS, ION_COLS))


def process_candidate(
    raw: str,
    scan: int,
    key_seq: str,
    ms2_map: Dict[int, Tuple[np.ndarray, np.ndarray, float, float]],
    msinfo: Dict[Tuple[str, int], str],
    fraginfo: Dict[Tuple[str, int], str],
) -> Tuple[np.ndarray, float, str, str, float]:
    """
    处理单个 target/decoy 候选：
    - key_seq: DeepFLR 编码序列（含 1/2/3/4）
    - raw, scan: 对应 Raw file 与 Scan number（Spectrum）
    - ms2_map: 当前 raw 的 MS2 谱图缓存
    - msinfo: (raw,scan) -> Mass analyzer
    - fraginfo: (raw,scan) -> Fragmentation

    返回：
    - train_mat: (31,29) 的强度矩阵
    - rt: 保留时间（若不可用则为 NaN）
    - analyzer: 质量分析器类型（来自 msms.txt）
    - frag: 碎裂方式（来自 msms.txt，如 HCD/CID；若不可用则为空字符串）
    - ce: 碰撞能量（来自 mzML precursor meta，若不可用则为 NaN）
    """
    zero_train = np.zeros((ION_ROWS, ION_COLS), dtype=float)
    default_analyzer = "FTMS"
    default_frag = ""
    default_ce = float("nan")

    # 1) key -> ProForma 风格 annotate（与 Mamba 输入保持一致）
    annotate = convert_deepflr_to_mamba_sequence(key_seq)

    # 2) 理论碎片（使用带缓存的实现，避免重复计算）
    theory_list = cached_process_single(annotate)
    if not theory_list:
        # 无理论碎片，直接返回全零
        return zero_train, float("nan"), default_analyzer, default_frag, default_ce

    # 3) 获取实验谱图
    mz_int_rt_ce = ms2_map.get(scan)
    if mz_int_rt_ce is None:
        # 找不到对应 MS2 谱图，返回全零
        return zero_train, float("nan"), default_analyzer, default_frag, default_ce

    mz_array, inten_array, rt, ce = mz_int_rt_ce

    # 4) 选择 Mass analyzer，决定 ppm/Da 容差
    analyzer = msinfo.get((raw, scan), default_analyzer)
    frag = fraginfo.get((raw, scan), default_frag)

    # 5) 强度匹配
    matched = fast_intensity_matching(
        theory_list,
        mz_array,
        inten_array,
        analyzer,
    )

    if matched is None:
        # 不支持的 analyzer（非 FTMS/ITMS），统一返回零矩阵，但仍返回元数据
        return zero_train, rt, analyzer, frag, ce

    # 6) 映射到固定网格
    ion_int_pairs = [(str(n), float(v)) for n, v in matched]
    train_mat = build_train_matrix(ion_int_pairs)
    return train_mat, rt, analyzer, frag, ce


# ---------------------------------------------------------------------
# 多进程并行：单个 Raw file 处理函数
# ---------------------------------------------------------------------

def _process_one_raw_worker(args: Tuple[str, List[int], str, Dict[Tuple[str, int], str], Dict[Tuple[str, int], str], pd.DataFrame]) -> Dict[str, Any]:
    """
    多进程 worker：处理单个 Raw file 的所有候选。
    
    参数（通过 tuple 传入以支持 Pool.imap）：
    - raw: Raw file 名称
    - idx_list: 该 raw 对应的所有候选行索引
    - mzml_dir: mzML 文件目录
    - msinfo: (raw, scan) -> Mass analyzer 映射
    - fraginfo: (raw, scan) -> Fragmentation 映射
    - td_subset: 该 raw 对应的 DataFrame 子集（包含 idx、Spectrum、key、PP.Charge 列）
    
    返回：
    - 包含处理结果的字典，key 为 idx，value 为结果元组
    """
    raw, idx_list, mzml_dir, msinfo, fraginfo, td_subset = args
    
    results: Dict[int, Tuple[np.ndarray, np.ndarray, float, str, str, float]] = {}
    
    # 加载 mzML
    mzml_path = os.path.join(mzml_dir, f"{raw}.mzML")
    try:
        ms2_map = load_mzml_ms2_map(mzml_path)
    except FileNotFoundError:
        ms2_map = {}
    
    # 处理该 raw 的所有候选
    for _, row in td_subset.iterrows():
        idx = int(row["_idx"])
        scan = row["Spectrum"]
        key_seq = row["key"]
        charge = row["PP.Charge"]
        
        if pd.isna(scan) or not key_seq:
            # 无合法 scan 或序列，填零
            train_mat = np.zeros((ION_ROWS, ION_COLS), dtype=float)
            by_mask = np.ones((29, 31), dtype=np.uint8)
            results[idx] = (train_mat, by_mask, float("nan"), "", "", float("nan"))
            continue
        
        train_mat, rt, analyzer, frag, ce = process_candidate(
            raw=str(raw),
            scan=int(scan),
            key_seq=str(key_seq),
            ms2_map=ms2_map,
            msinfo=msinfo,
            fraginfo=fraginfo,
        )
        
        # 计算 b/y 优先 mask（基于序列和电荷）
        annotate = convert_deepflr_to_mamba_sequence(str(key_seq))
        charge_val = int(charge) if not pd.isna(charge) else 2
        by_mask = compute_by_priority_mask_for_sequence(annotate, charge_val)
        
        results[idx] = (train_mat, by_mask, rt, analyzer, frag, ce)
    
    return {"raw": raw, "results": results}


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------

def build_ref_h5(
    target_decoy_csv: str,
    msms_path: str,
    mzml_dir: str,
    output_h5: str,
    quiet: bool = False,
    num_workers: int = 1,
) -> None:
    """
    核心入口：
    - 逐 raw 加载 mzML，并对该 raw 对应的所有 target/decoy 候选生成 train_data
    - 输出为候选级 H5，行数与 target_decoy CSV 完全一致（保证可与 Mamba 输入一一对应）
    """
    if not os.path.isfile(target_decoy_csv):
        raise FileNotFoundError(f"找不到 target_decoy 文件: {target_decoy_csv}")
    if not os.path.isfile(msms_path):
        raise FileNotFoundError(f"找不到 msms.txt 文件: {msms_path}")
    if not os.path.isdir(mzml_dir):
        raise NotADirectoryError(f"mzML 目录不存在: {mzml_dir}")

    td = pd.read_csv(target_decoy_csv)

    required_cols = ["SourceFile", "Spectrum", "PP.Charge", "key", "exp_strip_sequence"]
    for col in required_cols:
        if col not in td.columns:
            raise ValueError(f"target_decoy 文件缺少必要列: {col}")

    # 统一类型
    td["SourceFile"] = td["SourceFile"].astype(str)
    td["Spectrum"] = pd.to_numeric(td["Spectrum"], errors="coerce").astype("Int64")
    td["PP.Charge"] = pd.to_numeric(td["PP.Charge"], errors="coerce").astype("Int64")
    td["key"] = td["key"].astype(str)
    td["exp_strip_sequence"] = td["exp_strip_sequence"].astype(str)

    # 加载 msms 的 Mass analyzer / Fragmentation 信息
    msinfo, fraginfo = load_msms_msinfo(msms_path)

    n_rows = len(td)
    if not quiet:
        print(f"[INFO] target_decoy 候选总数: {n_rows}")
    else:
        # 精简模式也保留关键数量信息
        print(f"target_decoy 候选总数: {n_rows}")

    # 预分配结果容器（保持与 td 行顺序完全一致）
    train_mats: List[np.ndarray] = [None] * n_rows  # type: ignore
    by_priority_masks: List[np.ndarray] = [None] * n_rows  # type: ignore  # b/y 优先 mask
    rt_list: List[float] = [float("nan")] * n_rows
    analyzer_list: List[str] = ["" for _ in range(n_rows)]
    frag_list: List[str] = ["" for _ in range(n_rows)]
    ce_list: List[float] = [float("nan")] * n_rows

    # 按 SourceFile 分组，方便复用 mzML
    grouped_indices: Dict[str, List[int]] = {}
    for idx, raw in enumerate(td["SourceFile"]):
        if pd.isna(raw):
            continue
        grouped_indices.setdefault(str(raw), []).append(idx)

    # 为每个 raw 准备 DataFrame 子集（包含 _idx 列用于结果回填）
    td["_idx"] = td.index
    
    # 构建多进程任务参数列表
    task_args_list = []
    for raw, idx_list in grouped_indices.items():
        td_subset = td.loc[idx_list, ["_idx", "Spectrum", "key", "PP.Charge"]].copy()
        task_args_list.append((raw, idx_list, mzml_dir, msinfo, fraginfo, td_subset))
    
    # 根据 num_workers 选择串行或并行处理
    if num_workers <= 1:
        # 串行处理（保持原有逻辑，兼容单进程场景）
        if not quiet:
            print(f"[INFO] 使用串行模式处理 {len(grouped_indices)} 个 Raw file")
        iterator = task_args_list
        if not quiet:
            iterator = tqdm(iterator, desc="按 Raw file 处理候选", unit="raw")
        for args in iterator:
            result = _process_one_raw_worker(args)
            # 回填结果
            for idx, (train_mat, by_mask, rt, analyzer, frag, ce) in result["results"].items():
                train_mats[idx] = train_mat
                by_priority_masks[idx] = by_mask
                rt_list[idx] = float(rt) if not np.isnan(rt) else float("nan")
                analyzer_list[idx] = str(analyzer)
                frag_list[idx] = str(frag)
                ce_list[idx] = float(ce) if not np.isnan(ce) else float("nan")
            gc.collect()
    else:
        # 多进程并行处理
        if not quiet:
            print(f"[INFO] 使用 {num_workers} 进程并行处理 {len(grouped_indices)} 个 Raw file")
        
        with mp.Pool(processes=num_workers) as pool:
            if not quiet:
                results_iter = tqdm(
                    pool.imap_unordered(_process_one_raw_worker, task_args_list),
                    total=len(task_args_list),
                    desc="按 Raw file 并行处理",
                    unit="raw"
                )
            else:
                results_iter = pool.imap_unordered(_process_one_raw_worker, task_args_list)
            
            # 收集并回填结果
            for result in results_iter:
                for idx, (train_mat, by_mask, rt, analyzer, frag, ce) in result["results"].items():
                    train_mats[idx] = train_mat
                    by_priority_masks[idx] = by_mask
                    rt_list[idx] = float(rt) if not np.isnan(rt) else float("nan")
                    analyzer_list[idx] = str(analyzer)
                    frag_list[idx] = str(frag)
                    ce_list[idx] = float(ce) if not np.isnan(ce) else float("nan")
        
        gc.collect()

    # 将仍为 None 的位置填充为零矩阵/全 1 mask，确保不会出错
    for i in range(n_rows):
        if train_mats[i] is None:
            train_mats[i] = np.zeros((ION_ROWS, ION_COLS), dtype=float)
        if by_priority_masks[i] is None:
            by_priority_masks[i] = np.ones((29, 31), dtype=np.uint8)

    train_array = np.stack(train_mats, axis=0)  # (N, 31, 29)
    # 与 extract_real_spectrums.step4_merge_final_data 一致：n h w -> n w h
    train_array = np.swapaxes(train_array, 1, 2)  # (N, 29, 31)
    
    # b/y 优先 mask 数组：(N, 29, 31)
    by_priority_mask_array = np.stack(by_priority_masks, axis=0)

    # 元数据：Mass analyzer / Fragmentation / collision_energy
    analyzers = np.asarray(analyzer_list, dtype="S32")
    frags = np.asarray(frag_list, dtype="S16")
    # 若碰撞能量缺失，保持为 NaN，后续可根据需要再做填充
    collision_energies = np.array(ce_list, dtype=np.float32)

    # 写出 H5
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    raw_files = td["SourceFile"].astype(str).to_numpy(dtype="S100")
    scans = td["Spectrum"].astype("int64").to_numpy()
    charges = td["PP.Charge"].astype("int64").to_numpy()
    sequences = td["exp_strip_sequence"].astype(str).to_numpy(dtype="S128")
    keys = td["key"].astype(str).to_numpy(dtype="S256")
    rts = np.array(rt_list, dtype=np.float32)

    with h5py.File(output_h5, "w") as f:
        dset = f.create_dataset("Raw_file", data=raw_files)
        dset.attrs["description"] = "Raw file 名称（与 mamba 输入一一对应）"

        dset = f.create_dataset("MS2_Scan_Number", data=scans)
        dset.attrs["description"] = "MS2 扫描号（来自 msms.txt 的 Scan number）"

        dset = f.create_dataset("Charge", data=charges)
        dset.attrs["description"] = "肽段电荷状态"

        dset = f.create_dataset("Sequence", data=sequences)
        dset.attrs["description"] = "去修饰氨基酸序列（exp_strip_sequence）"

        dset = f.create_dataset("key", data=keys)
        dset.attrs["description"] = "DeepFLR 编码序列（含 1/2/3/4 的原始 key）"

        dset = f.create_dataset("RT", data=rts)
        dset.attrs["description"] = "保留时间（若不可用则为 NaN）"

        dset = f.create_dataset("Mass_analyzer", data=analyzers)
        dset.attrs["description"] = "质量分析器类型（来自 msms.txt 的 Mass analyzer 列）"

        dset = f.create_dataset("Fragmentation", data=frags)
        dset.attrs["description"] = "碎裂方式（来自 msms.txt 的 Fragmentation 列，如 HCD/CID，若不存在则为空）"

        dset = f.create_dataset("collision_energy", data=collision_energies)
        dset.attrs["description"] = "碰撞能量（来自 mzML precursor meta 的 collision energy，若缺失则为 NaN）"

        dset = f.create_dataset("train_data", data=train_array)
        dset.attrs["description"] = "候选级真实谱图强度矩阵，形状 (N, 29, 31)"

        dset = f.create_dataset("by_priority_mask", data=by_priority_mask_array)
        dset.attrs["description"] = "b/y 优先 mask，形状 (N, 29, 31)，1 表示保留，0 表示 m 离子与 b/y 质量冲突需置零"

        f.attrs["description"] = "基于 target_decoy 生成的候选级参考谱图 H5"
        f.attrs["source_target_decoy"] = os.path.abspath(target_decoy_csv)
        f.attrs["source_msms"] = os.path.abspath(msms_path)
        f.attrs["mzml_dir"] = os.path.abspath(mzml_dir)

    if not quiet:
        print(f"[OK] 已生成候选级参考谱图 H5: {output_h5}")
    # 精简模式与正常模式都输出形状信息
    print(f"train_data 形状: {train_array.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 DeepFLR step1_target_decoy.csv + msms.txt + mzML 构建候选级参考谱图 H5"
    )
    parser.add_argument(
        "--target_decoy_csv",
        required=True,
        help="DeepFLR1_generate_target_decoy.py 生成的 step1_target_decoy.csv 路径",
    )
    parser.add_argument(
        "--msms",
        required=True,
        help="原始 MaxQuant msms.txt 文件路径",
    )
    parser.add_argument(
        "--mzml-dir",
        required=True,
        help="mzML 文件所在目录（文件名需与 msms.txt 的 Raw file 对应，如 raw1.mzML）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 H5 文件路径（建议放在 rescore/origin_data_td.h5 或 origin_data.h5）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="精简输出，仅打印关键数量信息（关闭 tqdm 与大多数提示）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="并行处理的进程数（默认为 32；设置为 1 则使用串行模式）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_ref_h5(
        target_decoy_csv=args.target_decoy_csv,
        msms_path=args.msms,
        mzml_dir=args.mzml_dir,
        output_h5=args.output,
        quiet=args.quiet,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
