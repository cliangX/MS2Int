import gc
import multiprocessing as mp
import multiprocessing.dummy as mp_thread
import os
import warnings
from functools import lru_cache
import re

# 一些集群/网络文件系统默认启用 HDF5 文件锁，可能导致 to_hdf 在并发或 NFS 场景下报错：
# “unable to lock file, errno = 5, Input/output error”
# 必须在首次导入使用 HDF5 的库（pandas/PyTables）之前关闭文件锁。
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

# 说明：
# Step2 依赖 pyopenms 读取 mzML、依赖 spectrum_utils 生成理论碎片。
# 但后续单元测试只需要用到强度匹配函数（不需要这两个依赖）。
# 因此这里做可选导入：缺少依赖时允许模块被导入，并在真正调用相关功能时给出清晰报错。
try:
    import pyopenms as oms  # type: ignore
except Exception:
    oms = None  # type: ignore

try:
    from spectrum_utils import fragment_annotation, proforma  # type: ignore
except Exception:
    fragment_annotation = None  # type: ignore
    proforma = None  # type: ignore
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# 定义全局修饰转换字典（保持与调试脚本一致）
mod_transform = {
    r"(Oxidation (M))": "[Oxidation]",
    r"(Acetyl (Protein N-term))": "[Acetyl]-",
    r"(ac)": "[Acetyl]-",
    r"(ox)": "[Oxidation]",
    r"(Deamidation (NQ))": "[Deamidated]",
    "C": "C[Carbamidomethyl]",
    r"(de)": "[Deamidated]",
    "_": "",
    # 经典 STY 磷酸修饰（原 DeepFLR 数据）
    r"(Phospho (STY))": "[Phospho]",
    # Y 特异性磷酸修饰（如 Phospho(Y) 数据集）
    r"(Phospho(Y))": "[Phospho]",
    r"(Phospho (Y))": "[Phospho]",
}


def apply_modifications(sequence: str) -> str:
    """
    将 MaxQuant 风格的修饰名转换成 ProForma 所需的形式。
    这里仍然是简单的字符串替换，逻辑和调试脚本保持一致，并对非字符串输入做兜底转换。
    """
    if not isinstance(sequence, str):
        sequence = str(sequence)
    for key, value in mod_transform.items():
        sequence = sequence.replace(key, value)
    return sequence


# -----------------------------
# 理论碎片生成（单条序列，含磷酸位点感知的中性丢失过滤）
# -----------------------------
def generate_theoretical_fragments(annotate: str):
    """
    对单条 ProForma 序列生成理论碎片。

    关键逻辑：
    1. 解析 ProForma，获取序列长度与修饰信息；
    2. 自动识别磷酸化位点（Phospho），得到一组 1-based 位点编号；
    3. 对带 -H3PO4 的中性丢失碎片，仅在“覆盖到至少一个磷酸位点”的情况下保留；
    4. 其他碎片（包括 m 系列 2+ 离子）全部保留，交由后续 Step3 做统一归一化。

    返回：
        List[(fragment_name: str, mz: float)]
    """
    if proforma is None or fragment_annotation is None:
        raise ImportError(
            "缺少依赖 spectrum_utils（fragment_annotation/proforma），无法生成理论碎片；"
            "请先安装依赖后再运行 Step2。"
        )
    try:
        seq = proforma.parse(annotate)
    except Exception as e:
        print(f"Warning: ProForma 解析失败: {annotate} -> {e}")
        return []

    if not seq:
        return []

    try:
        theoretical_fragments = fragment_annotation.get_theoretical_fragments(
            seq[0],
            ion_types="byIm",
            max_charge=2,
            neutral_losses={
                # 中性丢失：磷酸
                "H3PO4": -97.976896,
            },
        )
    except Exception as e:
        print(f"Warning: 理论碎片生成失败: {annotate} -> {e}")
        return []

    # 序列长度与磷酸位点解析
    seq_len = len(seq[0].sequence)
    phospho_sites = set()

    # 优先使用 ProForma 解析出的修饰对象
    for mod in getattr(seq[0], "modifications", []) or []:
        pos = getattr(mod, "position", None)
        if not isinstance(pos, int):
            continue
        labels = [
            getattr(mod, "name", None),
            getattr(mod, "cv_entry", None),
            getattr(mod, "cv_label", None),
            str(mod),
        ]
        label_text = " ".join(str(x) for x in labels if x).lower()
        if "phospho" in label_text:
            # ProForma 位置是 0-based，这里统一转为 1-based
            phospho_sites.add(pos + 1)

    # 兜底：若解析不到修饰（兼容旧版 spectrum_utils），手动扫描 annotate 字符串
    if not phospho_sites:
        depth = 0
        pos = 0
        last_residue = None
        i = 0
        while i < len(annotate):
            ch = annotate[i]
            if ch == "[":
                depth += 1
                # 在进入修饰括号时检测是否为 [Phospho]
                if depth == 1 and last_residue is not None:
                    if annotate.startswith("[Phospho]", i):
                        phospho_sites.add(last_residue)
                i += 1
                continue
            if ch == "]":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and ch.isalpha() and ch.isupper():
                pos += 1
                last_residue = pos
            i += 1

    def covers_phospho(base_name: str) -> bool:
        """
        判断不带中性丢失后缀的碎片是否覆盖磷酸位点。
        - a/b/c/x/y/z 使用前/后缀覆盖区间；
        - mX:Y 内部碎片使用 [start, end) 区间；
        - immonium (I*) 无位置信息，默认不允许中性丢失。
        """
        if not phospho_sites:
            return False

        # 先移除末尾的电荷标识（如 y7^2 -> y7）
        base_no_charge = re.sub(r"\^\d+$", "", base_name)

        # 内部碎片 mX:Y 覆盖 [start, end) 的 1-based 区间
        m_match = re.match(r"^m(\d+):(\d+)", base_no_charge)
        if m_match:
            start = int(m_match.group(1))
            end = int(m_match.group(2))
            covered = set(range(start, end))
            return bool(phospho_sites & covered)

        # immonium 离子无位置信息，直接拒绝中性丢失
        if base_name.startswith("I"):
            return False

        # 常规前缀/后缀离子
        match = re.match(r"^([abcxyz])(\d+)", base_no_charge)
        if not match:
            return False

        ion_type, idx_str = match.groups()
        idx = int(idx_str)
        if ion_type in "abc":
            covered = set(range(1, idx + 1))
        else:  # x/y/z
            start = max(1, seq_len - idx + 1)
            covered = set(range(start, seq_len + 1))

        return bool(phospho_sites & covered)

    result = []
    for fragment, value in theoretical_fragments:
        name = str(fragment)

        # 对带 -H3PO4 的中性丢失碎片，仅保留理论上覆盖磷酸位点的情况
        if "-H3PO4" in name:
            check_name = name.replace("-H3PO4", "")
            if not covers_phospho(check_name):
                continue

        # 其余情况（包括 m 系列 2+ 离子）全部保留，
        # 后续在 Step3 中再通过 _normalize_ion_name / ION_TO_IDX 进行统一归一化和合并。
        result.append((name, value))

    return result


# 全局缓存函数（在每个进程中独立缓存）
@lru_cache(maxsize=15)
def cached_process_single(annotate):
    """带缓存的单个序列处理，内部调用带磷酸位点感知逻辑的理论碎片生成函数。"""
    frags = generate_theoretical_fragments(annotate)
    # 为了兼容下游逻辑，这里仍然返回 tuple 形式
    return tuple(frags)


def process_batch(annotate_batch):
    """处理一批序列（在单个进程中）"""
    results = []
    for annotate in annotate_batch:
        result = cached_process_single(annotate)
        results.append(list(result))  # 转回list格式
    return results


def parallel_process_with_cache(
    annotates, num_processes=4, batch_size=1000, prefer_threads=None, verbose=False
):
    """多进程+缓存+批处理，并带进度条"""
    batches = []
    for i in range(0, len(annotates), batch_size):
        batch = annotates[i : i + batch_size]
        batches.append(batch)

    if verbose:
        print(f"Processing {len(annotates)} sequences in {len(batches)} batches")

    batch_results = []
    use_threads = (
        prefer_threads if prefer_threads is not None else mp.current_process().daemon
    )
    _pool_mod = mp_thread if use_threads else mp
    with _pool_mod.Pool(processes=num_processes) as pool:
        iterator = pool.imap(process_batch, batches)
        if verbose:
            iterator = tqdm(iterator, total=len(batches), desc="theoretical fragments")
        for res in iterator:
            batch_results.append(res)

    final_results = []
    for batch_result in batch_results:
        final_results.extend(batch_result)

    return final_results


# -----------------------------
# Parallel intensity matching
# -----------------------------

# 尝试引入 numba，加速窗口搜索与最大值计算
try:
    from numba import njit  # type: ignore

    _NUMBA = True
except Exception:
    _NUMBA = False


def _match_fragments_python(
    theory_mz: np.ndarray,
    mz_sorted: np.ndarray,
    inten_sorted: np.ndarray,
    tol_value: float,
    is_ppm: bool,
) -> np.ndarray:
    out = np.zeros(theory_mz.shape[0], dtype=np.float64)
    for idx in range(theory_mz.shape[0]):
        tmz = theory_mz[idx]
        if np.isnan(tmz):
            out[idx] = 0.0
            continue
        tol = tmz * tol_value * 1e-6 if is_ppm else tol_value
        left = np.searchsorted(mz_sorted, tmz - tol, side="left")
        right = np.searchsorted(mz_sorted, tmz + tol, side="right")
        if right > left:
            # 手动遍历求最大，避免在极短窗口频繁创建临时切片
            max_val = inten_sorted[left]
            for j in range(left + 1, right):
                if inten_sorted[j] > max_val:
                    max_val = inten_sorted[j]
            out[idx] = float(max_val)
        else:
            out[idx] = 0.0
    return out


if _NUMBA:

    @njit(cache=True, fastmath=True)
    def _match_fragments_numba(theory_mz, mz_sorted, inten_sorted, tol_value, is_ppm):
        out = np.zeros(theory_mz.shape[0], dtype=np.float64)
        for idx in range(theory_mz.shape[0]):
            tmz = theory_mz[idx]
            if np.isnan(tmz):
                out[idx] = 0.0
                continue
            tol = tmz * tol_value * 1e-6 if is_ppm else tol_value
            left = np.searchsorted(mz_sorted, tmz - tol, side="left")
            right = np.searchsorted(mz_sorted, tmz + tol, side="right")
            if right > left:
                max_val = inten_sorted[left]
                for j in range(left + 1, right):
                    if inten_sorted[j] > max_val:
                        max_val = inten_sorted[j]
                out[idx] = max_val
            else:
                out[idx] = 0.0
        return out
else:
    _match_fragments_numba = _match_fragments_python  # noqa: E305


def fast_intensity_matching(
    theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer
):
    """Match theoretical m/z to experimental peaks and take max intensity within tolerance.

    Returns a list like [[frag_str, matched_intensity], ...]. If mass_analyzer is
    not FTMS or ITMS, return None to mimic previous behavior (skip row).
    """
    if theory_mz_list is None:
        return None

    # Keep previous semantics: skip rows with unknown analyzer
    if mass_analyzer not in ("FTMS", "ITMS"):
        return None

    # Prepare arrays
    if experiment_mz_list is None or experiment_int_list is None:
        # No experimental data; return zeros
        return [[val[0], 0.0] for val in theory_mz_list]

    mz = np.asarray(experiment_mz_list, dtype=float)
    inten = np.asarray(experiment_int_list, dtype=float)
    if mz.size == 0:
        return [[val[0], 0.0] for val in theory_mz_list]

    # Ensure sorted by m/z for binary searches
    if mz.ndim != 1:
        mz = mz.ravel()
    if inten.ndim != 1:
        inten = inten.ravel()
    order = np.argsort(mz)
    mz_sorted = mz[order]
    inten_sorted = inten[order]

    # 构造理论m/z数组（None -> NaN）
    names = []
    mz_vals = []
    # by 优先规则的预处理：
    # - 收集 b/y 理论 m/z；
    # - 记录 m 碎片在 theory_mz_list 中的索引；
    # - 若某个 m 的理论 m/z 在“自身容差窗口”内与任一 b/y 理论 m/z 重合，则该 m 直接置 0（后续置 NaN 即可）。
    by_mz_vals = []
    m_indices = []

    for idx, (frag_str, tmz) in enumerate(theory_mz_list):
        names.append(frag_str)
        mz_val = np.nan if tmz is None else float(tmz)
        mz_vals.append(mz_val)

        if tmz is None or frag_str is None:
            continue

        # 仅把 b/y 作为“by 区域”；不将 immonium（I*）纳入 by 去重集合。
        # frag_str 常见形式：b7, b7^2, y9, y9^2, b7-H3PO4, y9-H3PO4^2 等
        if (
            isinstance(frag_str, str)
            and len(frag_str) >= 2
            and frag_str[0] in ("b", "y")
            and frag_str[1].isdigit()
        ):
            if not np.isnan(mz_val):
                by_mz_vals.append(mz_val)
        elif isinstance(frag_str, str) and frag_str.startswith("m"):
            m_indices.append(idx)
    theory_mz = np.asarray(mz_vals, dtype=np.float64)

    # 容差：FTMS用20ppm，否则0.5 Da
    is_ppm = True if mass_analyzer == "FTMS" else False
    tol_value = 20.0 if is_ppm else 0.5

    # 应用 by 优先去重：将冲突的 m 碎片标记为 NaN（后续匹配会输出 0）
    if by_mz_vals and m_indices:
        by_mz_sorted = np.sort(np.asarray(by_mz_vals, dtype=np.float64))
        for mi in m_indices:
            tmz = theory_mz[mi]
            if np.isnan(tmz):
                continue
            tol = tmz * tol_value * 1e-6 if is_ppm else tol_value
            left = np.searchsorted(by_mz_sorted, tmz - tol, side="left")
            right = np.searchsorted(by_mz_sorted, tmz + tol, side="right")
            if right > left:
                theory_mz[mi] = np.nan

    intensities = _match_fragments_numba(
        theory_mz, mz_sorted, inten_sorted, tol_value, is_ppm
    )

    out = []
    for i, name in enumerate(names):
        out.append([name, float(intensities[i])])
    return out


def process_single_spectrum(args):
    """Worker: process one spectrum tuple."""
    theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer = args
    return fast_intensity_matching(
        theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer
    )


def parallel_intensity_matching(
    combined_df, num_processes=4, batch_size=1000, prefer_threads=None, verbose=False
):
    """单进程池 + 流式imap 的强度匹配实现。

    - 单个进程池贯穿整个阶段，避免每批次反复创建/销毁进程。
    - 使用 chunksize=batch_size 控制任务分发粒度（推荐500）。
    - 返回与 combined_df 行对齐的结果列表。
    """

    args_list = list(
        zip(
            combined_df["theoretical_fragments"],
            combined_df["mzarray"],
            combined_df["intarray"],
            combined_df["Mass analyzer"],
        )
    )

    use_threads = (
        prefer_threads if prefer_threads is not None else mp.current_process().daemon
    )
    _pool_mod = mp_thread if use_threads else mp

    chunksize = max(1, int(batch_size))

    with _pool_mod.Pool(processes=num_processes) as pool:
        iterator = pool.imap(process_single_spectrum, args_list, chunksize=chunksize)
        if verbose:
            iterator = tqdm(iterator, total=len(args_list), desc="intensity matching")
        results = list(iterator)

    return results


# 将process_pair函数移到外部
def process_pair(meta_path, mz_path, msms_root, df_h5_dir, inner_num_procs=4):
    try:
        if oms is None:
            raise ImportError(
                "缺少依赖 pyopenms，无法读取 mzML；请先安装依赖后再运行 Step2。"
            )
        # 获取文件名，用于构建MSMS文件路径
        file_name = os.path.basename(meta_path)

        # 构建MSMS文件路径
        MSMS = os.path.join(msms_root, file_name)

        df1 = pd.read_csv(MSMS, sep="\t", low_memory=False)
        # df = pd.read_csv(meta_path, sep="\t",low_memory=False)
        df = pd.read_csv(MSMS, sep="\t", low_memory=False)

        # 过滤和重命名列
        df_filtered = pd.read_csv(MSMS, sep="\t", low_memory=False)
        columns_to_keep = [
            "Sequence",
            "Length",
            "Modifications",
            "Modified sequence",
            "Charge",
            "Scan number",
            "Score",
            "Raw file",
            "Reverse",
        ]
        new_column_names = [
            "Sequence",
            "Length",
            "Modifications",
            "Modified_sequence",
            "Charge",
            "MS2_Scan_Number",
            "Score",
            "Raw_file",
            "Reverse",
        ]
        meta_df = df_filtered[columns_to_keep]
        meta_df.columns = new_column_names

        # 应用修改
        meta_df["annotate"] = meta_df["Modified_sequence"].apply(apply_modifications)

        # 新建 'Mass_analyzer' 列并匹配填充
        # 确保 MS2_Scan_Number 和 Scan number 是相同的数据类型，通常都应为整型
        df1["Scan number"] = df1["Scan number"].astype(int)
        meta_df["MS2_Scan_Number"] = meta_df["MS2_Scan_Number"].astype(int)

        # 将 df1 中的 'Mass analyzer' 信息根据 'Scan number' 匹配到 meta_df 中
        meta_df = meta_df.merge(
            df1[["Scan number", "Mass analyzer", "Fragmentation"]],
            left_on="MS2_Scan_Number",
            right_on="Scan number",
            how="left",
        )

        # 删除不再需要的 'Scan number' 列
        meta_df.drop(columns=["Scan number"], inplace=True)

        ## 打注释标签（多进程 + 缓存 + 批处理 + 进度条）
        # 这里在部分环境下使用多进程 (multiprocessing) 会触发
        # UnexpectedEOF 相关的 Pickle 错误，因此改为优先使用线程池，
        # 避免跨进程序列化问题。
        meta_df["theoretical_fragments"] = parallel_process_with_cache(
            meta_df["annotate"].values.tolist(),
            num_processes=inner_num_procs,
            batch_size=500,
            prefer_threads=True,
            verbose=False,
        )

        # 处理 .mzML 文件
        exp = oms.MSExperiment()
        oms.MzMLFile().load(mz_path, exp)

        normalizer = oms.Normalizer()
        param = normalizer.getParameters()
        param.setValue("method", "to_one")
        normalizer.setParameters(param)
        normalizer.filterPeakMap(exp)

        mz_df = exp.get_df()
        mz_df["instrument"] = exp.getExperimentalSettings().getInstrument().getName()

        mz_df["collision_energy"] = None
        mz_df["MS2_Scan_Number"] = None
        mz_df["Fragmentation_mzml"] = None
        for idx, spectrum in enumerate(exp):
            if spectrum.getMSLevel() == 2 and spectrum.getPrecursors():
                precursor = spectrum.getPrecursors()[0]
                mz_df.at[idx, "MS2_Scan_Number"] = idx + 1
                if precursor.metaValueExists("collision energy"):
                    collision_energy = precursor.getMetaValue("collision energy")
                    mz_df.at[idx, "collision_energy"] = collision_energy

                # 从 mzML 的 precursor activation method 提取 Fragmentation（短字符串，如 "HCD"/"CID"）
                try:
                    short = [
                        x.decode() for x in precursor.getActivationMethodsAsShortString()
                    ]  # 例如 ["HCD"]
                    frag = short[0] if short else ""
                except Exception:
                    frag = ""
                if frag:
                    mz_df.at[idx, "Fragmentation_mzml"] = frag

        combined_df = pd.merge(meta_df, mz_df, on="MS2_Scan_Number", how="left")
        if "Fragmentation_mzml" in combined_df.columns:
            mzml_frag = combined_df["Fragmentation_mzml"]
            use_mzml = mzml_frag.notna() & (mzml_frag.astype(str).str.strip() != "")
            if use_mzml.any():
                combined_df.loc[use_mzml, "Fragmentation"] = mzml_frag.loc[use_mzml]
                print(
                    f"Fragmentation检查: 使用mzML覆盖={int(use_mzml.sum())}/{len(combined_df)}"
                )
            combined_df.drop(columns=["Fragmentation_mzml"], inplace=True)
        combined_df["collision_energy"] = (
            combined_df["collision_energy"].astype(float).fillna(30)
        )
        # 检验collision_energy列的数据情况
        if "collision_energy" in combined_df.columns:
            null_count = combined_df["collision_energy"].isna().sum()
            total_count = len(combined_df)
            valid_count = total_count - null_count
            print(
                f"collision_energy检查: 总数={total_count}, 有效值={valid_count}, 空值={null_count}"
            )
            if valid_count > 0:
                valid_values = combined_df["collision_energy"].dropna()
                print(
                    f"有效collision_energy值范围: {valid_values.min():.2f} - {valid_values.max():.2f}"
                )
        else:
            print("警告: combined_df中没有collision_energy列")

        combined_df["theoretical_fragments_int"] = parallel_intensity_matching(
            combined_df,
            num_processes=inner_num_procs,
            batch_size=500,
            prefer_threads=True,
            verbose=False,
        )
        # 构造存储路径并保存 DataFrame
        output_path = os.path.join(df_h5_dir, meta_df["Raw_file"].iloc[0] + ".h5")
        combined_df.to_hdf(output_path, key="combined_data", mode="w")
        print(f"Processed and saved: {output_path}")
        gc.collect()
        return True
    except Exception as e:
        print(f"Error processing {meta_path} and {mz_path}: {str(e)}")
        return False


def run_step2(config):
    # 从配置中获取路径和性能参数
    msms_root = config["paths"]["msms_filtered_dir"]
    mzml_root = config["paths"]["mzml_dir"]
    search_root = config["paths"]["search_dir"]
    result_base_path = os.path.dirname(config["paths"]["df_h5_dir"])
    df_h5_dir = config["paths"]["df_h5_dir"]

    # 获取性能参数
    num_work = config["performance"]["num_workers"]

    # 创建结果文件夹（如果不存在）
    os.makedirs(df_h5_dir, exist_ok=True)

    # 定义Search目录路径
    directory_path = search_root

    # 获取目录下的所有文件和目录名
    files = os.listdir(directory_path)

    # 初始化列表以存放所有合并后的文件名部分
    experiment_names = []

    # 打印所有文件名，分割并合并前六个字符
    for file in files:
        # 使用下划线作为分隔符分割文件名
        split_name = file.split(".txt")
        # 合并分割后的前六个元素（如果存在）
        combined_parts = split_name[0]  # 只取前六个分割的部分
        # 将合并后的部分添加到列表中
        experiment_names.append(combined_parts)

    # 最后，打印出整个列表，以验证所有元素已正确添加
    print("All combined experiment names:")
    print(experiment_names[1:6] if len(experiment_names) > 5 else experiment_names)

    # 基于精确文件名（去扩展名）的配对
    # 构建 stem -> path 映射
    mzml_map = {}
    for file in os.listdir(mzml_root):
        if file.lower().endswith(".mzml"):
            stem = os.path.splitext(file)[0]
            mzml_map[stem] = os.path.join(mzml_root, file)

    search_map = {}
    for file in os.listdir(search_root):
        if file.endswith(".txt"):
            stem = os.path.splitext(file)[0]
            search_map[stem] = os.path.join(search_root, file)

    # 精确匹配 experiment_name 与 stem 完全相等
    valid_pairs = []
    valid_mzml_files = []
    valid_search_files = []

    for experiment_name in experiment_names:
        search_path = search_map.get(experiment_name)
        if search_path is None:
            print(f"Warning: No Search file found for exact name '{experiment_name}'")
            continue

        mzml_path = mzml_map.get(experiment_name)
        if mzml_path is None:
            print(f"Warning: No mzML file found for exact name '{experiment_name}'")
            continue

        valid_search_files.append(search_path)
        valid_mzml_files.append(mzml_path)
        valid_pairs.append((experiment_name, search_path, mzml_path))

    print(f"Found {len(valid_pairs)} valid file pairs to process")

    def main_process(search_files, mzml_files):
        # 外层串行 + 内层并行（32）
        inner_num_procs = 32
        print(f"使用串行文件处理 + 内层{inner_num_procs}核心并行计算")

        results = []
        # 串行处理每个文件对，使用tqdm显示进度
        for search_file, mzml_file in tqdm(
            zip(search_files, mzml_files), total=len(search_files), desc="处理文件"
        ):
            result = process_pair(
                search_file, mzml_file, msms_root, df_h5_dir, inner_num_procs
            )
            results.append(result)
            # 主动清理缓存，避免内存累积
            gc.collect()

        successful = sum(1 for r in results if r)
        print(f"Successfully processed {successful} out of {len(results)} file pairs")

    # 执行主处理函数
    main_process(valid_search_files, valid_mzml_files)

    print("3.2完成")


if __name__ == "__main__":
    # 仅用于直接运行此脚本的测试
    import os

    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step2(config)
