import gc
import multiprocessing as mp
import multiprocessing.dummy as mp_thread
import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pyopenms as oms
from spectrum_utils import fragment_annotation, proforma
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# 定义全局修饰转换字典
mod_transform = {
    r"(Oxidation (M))": "[Oxidation]",
    r"(Acetyl (Protein N-term))": "[Acetyl]-",
    r"(ac)": "[Acetyl]-",
    r"(ox)": "[Oxidation]",
    r"(Deamidation (NQ))": "[Deamidated]",
    "C": "C[Carbamidomethyl]",
    r"(de)": "[Deamidated]",
    "_": "",
}


# 将函数移到外部
def apply_modifications(sequence):
    for key, value in mod_transform.items():
        sequence = sequence.replace(key, value)
    return sequence


# 全局缓存函数（在每个进程中独立缓存）
@lru_cache(maxsize=15)
def cached_process_single(annotate):
    """带缓存的单个序列处理"""
    seq = proforma.parse(annotate)
    if not seq:
        return tuple()

    theoretical_fragments = fragment_annotation.get_theoretical_fragments(
        seq[0], ion_types="byIm", max_charge=2
    )

    result = tuple(
        (str(fragment), value)
        for fragment, value in theoretical_fragments
        if not ("m" in str(fragment) and "^" in str(fragment))
    )
    return result


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
    for frag_str, tmz in theory_mz_list:
        names.append(frag_str)
        mz_vals.append(np.nan if tmz is None else float(tmz))
    theory_mz = np.asarray(mz_vals, dtype=np.float64)

    # 容差：FTMS用20ppm，否则0.5 Da
    is_ppm = True if mass_analyzer == "FTMS" else False
    tol_value = 20.0 if is_ppm else 0.5

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
        meta_df["theoretical_fragments"] = parallel_process_with_cache(
            meta_df["annotate"].values.tolist(),
            num_processes=inner_num_procs,
            batch_size=500,
            prefer_threads=False,
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
            prefer_threads=False,
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

    # 存储找到的文件路径
    mzml_files = []
    search_files = []

    # 遍历experiment_names列表
    for experiment_name in experiment_names:
        # 在mzml_root目录中寻找包含experiment_name且扩展名为.mzML的文件的完整路径
        found_mzml = False
        for file in os.listdir(mzml_root):
            if experiment_name in file and file.lower().endswith(".mzml"):
                full_path = os.path.join(mzml_root, file)
                mzml_files.append(full_path)
                found_mzml = True
                break  # 假设每个experiment_name只对应一个文件，找到即跳出循环

        if not found_mzml:
            print(f"Warning: No mzML file found for {experiment_name}")

        # 在search_root目录中寻找包含experiment_name的文件的完整路径
        found_search = False
        for file in os.listdir(search_root):
            if experiment_name in file:
                full_path = os.path.join(search_root, file)
                search_files.append(full_path)
                found_search = True
                break  # 假设每个experiment_name只对应一个文件，找到即跳出循环

        if not found_search:
            print(f"Warning: No Search file found for {experiment_name}")

    # 确保数据对是完整的
    valid_pairs = []
    valid_mzml_files = []
    valid_search_files = []

    for i, (search_file, mzml_file) in enumerate(zip(search_files, mzml_files)):
        if i < len(search_files) and i < len(mzml_files):
            valid_mzml_files.append(mzml_file)
            valid_search_files.append(search_file)
            valid_pairs.append((experiment_names[i], search_file, mzml_file))

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
