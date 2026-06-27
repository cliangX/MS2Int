import os
import pandas as pd
import numpy as np
import warnings
import gc
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# 忽略 PerformanceWarning
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)

# -----------------------------
# 断点续跑 / 可读性检查工具函数
# -----------------------------


def _is_valid_combined_h5(path: str) -> bool:
    """判断 h5 是否可读（包含 combined_data 且结构完整）。

    说明：仅用是否存在 /combined_data 不够。某些中断/异常会留下结构不完整的文件，
    在读取时会报 `block*_items_variety 不存在` 之类的错误。
    这里用 PyTables 做快速结构校验（不读取完整 DataFrame）。
    """
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
    except OSError:
        return False

    try:
        import tables

        with tables.open_file(path, mode="r") as h5:
            if "/combined_data" not in h5:
                return False

            node = h5.get_node("/combined_data")
            children = set(getattr(node, "_v_children", {}).keys())

            for axis in ("axis0", "axis1"):
                if axis not in children:
                    return False
                if not hasattr(node._v_attrs, f"{axis}_variety"):
                    return False

            try:
                nblocks = int(getattr(node._v_attrs, "nblocks"))
            except Exception:
                nblocks = None

            if nblocks is not None:
                for i in range(nblocks):
                    items = f"block{i}_items"
                    values = f"block{i}_values"
                    if items not in children or values not in children:
                        return False
                    if not hasattr(node._v_attrs, f"{items}_variety"):
                        return False
            else:
                for name in children:
                    if name.startswith("block") and name.endswith("_items"):
                        if not hasattr(node._v_attrs, f"{name}_variety"):
                            return False
                        values = name.replace("_items", "_values")
                        if values not in children:
                            return False

        return True
    except Exception:
        return False


def _atomic_to_hdf(df: pd.DataFrame, output_path: str, key: str) -> None:
    """以临时文件写入 + 原子替换的方式落盘，避免中途中断留下损坏文件。"""
    tmp_path = output_path + ".tmp"
    try:
        df.to_hdf(tmp_path, key=key, mode="w")
        os.replace(tmp_path, output_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# -----------------------------
# Prebuild annotation matrix and index mapping (module-level)
# -----------------------------
MAX_PEPTIDE_LENGTH = 40
ION_ROWS = MAX_PEPTIDE_LENGTH + 1
ION_COLS = MAX_PEPTIDE_LENGTH - 1


def _build_annotation():
    """构建与最大肽段长度一致的注释矩阵和 name->flat-index 映射。

    去掉了原本用于存放亚胺离子(Immonium ions)的单独一列。
    现在直接以 (MAX_PEPTIDE_LENGTH - 1) 作为矩阵的列数。
    """
    mat = np.full((ION_ROWS, ION_COLS), "", dtype=object)
    cleavage_cols = ION_COLS
    for i in range(cleavage_cols):
        mat[0, i] = f"b{i + 1}"
    for i in range(cleavage_cols):
        mat[1, i] = f"b{i + 1}^2"
    for i in range(cleavage_cols):
        mat[2, i] = f"y{i + 1}"
    for i in range(cleavage_cols):
        mat[3, i] = f"y{i + 1}^2"
    for row in range(4, ION_ROWS):
        for col in range(cleavage_cols):
            m_start = col + 2
            m_end = m_start + (row - 2)
            if m_end <= MAX_PEPTIDE_LENGTH:
                mat[row, col] = f"m{m_start}:{m_end}"

    ion_order = mat.ravel().tolist()
    index = {name: i for i, name in enumerate(ion_order) if name}
    return mat, ion_order, index


ANNOTATION_MATRIX, ION_ORDER, ION_TO_IDX = _build_annotation()


def process_file(file_path, ylabel_df_dir):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(ylabel_df_dir, stem + ".h5")

    force = str(os.environ.get("FORCE", "0")).strip() == "1"
    if (not force) and _is_valid_combined_h5(output_path):
        return {"ok": True, "skipped": True, "input": file_path, "output": output_path}

    try:
        with pd.HDFStore(file_path, "r") as store:
            df = store["combined_data"]
    except Exception as e:
        return {
            "ok": False,
            "input": file_path,
            "error": f"读取失败: {type(e).__name__}: {e}",
        }

    try:
        if len(df) == 0:
            return {
                "ok": False,
                "input": file_path,
                "error": "读取成功但 combined_data 为空",
            }

        n = len(df)
        train_data = [None] * n
        flat_len = ION_ROWS * ION_COLS
        for idx in range(n):
            if pd.isna(df["Mass analyzer"].iat[idx]):
                continue
            fragments = df["theoretical_fragments_int"].iat[idx]
            if not fragments:
                train_data[idx] = np.zeros((ION_ROWS, ION_COLS), dtype=float)
                continue

            vec = np.zeros(flat_len, dtype=float)
            filled = np.zeros(flat_len, dtype=bool)
            try:
                for name, inten in fragments:
                    j = ION_TO_IDX.get(name)
                    if j is not None and not filled[j]:
                        vec[j] = float(inten)
                        filled[j] = True
            except Exception:
                pass
            train_data[idx] = vec.reshape((ION_ROWS, ION_COLS))

        df["train_data"] = train_data

        keep_cols = [
            "Sequence",
            "Length",
            "Modifications",
            "Modified_sequence",
            "Charge",
            "MS2_Scan_Number",
            "Score",
            "Raw_file",
            "annotate",
            "RT",
            "instrument",
            "collision_energy",
            "Mass analyzer",
            "Fragmentation",
            "Reverse",
            "train_data",
        ]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        _atomic_to_hdf(df, output_path, key="combined_data")
        return {"ok": True, "skipped": False, "input": file_path, "output": output_path}
    except Exception as e:
        return {
            "ok": False,
            "input": file_path,
            "error": f"处理/写入失败: {type(e).__name__}: {e}",
        }
    finally:
        gc.collect()


def run_step3(config):
    df_h5_dir = config["paths"]["df_h5_dir"]
    ylabel_df_dir = config["paths"]["ylabel_df_dir"]
    num_work = config["performance"]["num_workers"]

    os.makedirs(ylabel_df_dir, exist_ok=True)

    file_paths = [
        os.path.join(df_h5_dir, file)
        for file in os.listdir(df_h5_dir)
        if file.endswith(".h5")
    ]

    process_func = partial(process_file, ylabel_df_dir=ylabel_df_dir)

    with Pool(processes=num_work) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_func, file_paths, chunksize=1),
                total=len(file_paths),
                desc="Processing files",
                mininterval=0.5,
            )
        )
    gc.collect()

    ok_results = [r for r in results if isinstance(r, dict) and r.get("ok")]
    skipped = [r for r in ok_results if r.get("skipped")]
    failed = [r for r in results if isinstance(r, dict) and (not r.get("ok"))]

    print(f"[OK] Step3 输入文件数: {len(file_paths)}")
    print(f"[OK] 成功处理: {len(ok_results) - len(skipped)}")
    print(f"[SKIP] 已存在且可读: {len(skipped)}")
    print(f"[FAIL] 读取/处理/写入失败: {len(failed)}")

    if failed:
        report_path = os.path.join(ylabel_df_dir, "_step3_failed_inputs.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                for r in failed:
                    f.write(f"{r.get('input')}\t{r.get('error')}\n")
            print(f"[WARN] 失败清单已写入: {report_path}")
        except Exception as e:
            print(f"Warning: 写入失败清单失败 {report_path}: {e}")

    if len(ok_results) == 0:
        raise RuntimeError(
            "Step3 未生成任何有效输出，请先检查 Step2 的 df_h5_dir 是否为空或全部损坏。"
        )

    print("3.3完成")


if __name__ == "__main__":
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step3(config)
