import gc
import multiprocessing as mp
import multiprocessing.dummy as mp_thread
import os
import re
import sys
import warnings
from functools import lru_cache
from itertools import repeat

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

# 兼容旧版 numba cache 中记录的顶层模块名，避免多进程反序列化时导入失败。
sys.modules.setdefault("step2_process_df_h5", sys.modules[__name__])

try:
    import pyopenms as oms  # type: ignore
except ImportError:
    oms = None  # type: ignore

try:
    from spectrum_utils import fragment_annotation, proforma  # type: ignore
except ImportError:
    fragment_annotation = None  # type: ignore
    proforma = None  # type: ignore
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _is_valid_combined_h5(path: str) -> bool:
    """判断 step2 输出的 h5 是否完整可用。"""
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
        return True
    except Exception:
        try:
            with pd.HDFStore(path, mode="r") as store:
                return "/combined_data" in store.keys()
        except Exception:
            return False


def _atomic_to_hdf(df: pd.DataFrame, output_path: str, key: str) -> None:
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


mod_transform = {
    r"(Oxidation (M))": "[Oxidation]",
    r"(Acetyl (Protein N-term))": "[Acetyl]-",
    r"(ac)": "[Acetyl]-",
    r"(ox)": "[Oxidation]",
    r"(Deamidation (NQ))": "[Deamidated]",
    "C": "C[Carbamidomethyl]",
    r"(de)": "[Deamidated]",
    "_": "",
    r"(Phospho (STY))": "[Phospho]",
    r"(Phospho(Y))": "[Phospho]",
    r"(Phospho (Y))": "[Phospho]",
}


def apply_modifications(sequence):
    if not isinstance(sequence, str):
        sequence = str(sequence)
    for key, value in mod_transform.items():
        sequence = sequence.replace(key, value)
    return sequence


def generate_theoretical_fragments(annotate, mode: str):
    """Generate theoretical fragments for the given mode.

    - unmodified: use byIm/max_charge=2, filter out charged m ions (containing 'm' and '^').
    - phospho: add H3PO4 neutral loss on top of byIm/max_charge=2, keep only -H3PO4 fragments covering phosphosite.
    """
    if proforma is None or fragment_annotation is None:
        raise ImportError("Missing dependency: spectrum_utils (required for theoretical fragment generation)")

    mode = str(mode or "unmodified").strip().lower()
    annotate = str(annotate)

    try:
        seq = proforma.parse(annotate)
    except Exception as e:
        print(f"Warning: ProForma parse failed: {annotate} -> {e}")
        return []

    if not seq:
        return []

    if mode == "phospho":
        try:
            theoretical_fragments = fragment_annotation.get_theoretical_fragments(
                seq[0],
                ion_types="byIm",
                max_charge=2,
                neutral_losses={
                    "H3PO4": -97.976896,
                },
            )
        except Exception as e:
            print(f"Warning: theoretical fragment generation failed: {annotate} -> {e}")
            return []

        seq_len = len(seq[0].sequence)
        phospho_sites = set()

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
                phospho_sites.add(pos + 1)

        # Fallback: if modifications were not parsed, detect [Phospho] positions from the ProForma string directly
        if not phospho_sites:
            depth = 0
            pos = 0
            last_residue = None
            i = 0
            while i < len(annotate):
                ch = annotate[i]
                if ch == "[":
                    depth += 1
                    if depth == 1 and last_residue is not None:
                        if str(annotate).startswith("[Phospho]", i):
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
            if not phospho_sites:
                return False

            base_no_charge = re.sub(r"\^\d+$", "", base_name)

            m_match = re.match(r"^m(\d+):(\d+)", base_no_charge)
            if m_match:
                start = int(m_match.group(1))
                end = int(m_match.group(2))
                covered = set(range(start, end))
                return bool(phospho_sites & covered)

            if base_name.startswith("I"):
                return False

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
            if "-H3PO4" in name:
                check_name = name.replace("-H3PO4", "")
                if not covers_phospho(check_name):
                    continue
            result.append((name, value))
        return result

    theoretical_fragments = fragment_annotation.get_theoretical_fragments(
        seq[0], ion_types="byIm", max_charge=2
    )
    result = []
    for fragment, value in theoretical_fragments:
        name = str(fragment)
        if ("m" in name) and ("^" in name):
            continue
        result.append((name, value))
    return result


@lru_cache(maxsize=15)
def cached_process_single(annotate, mode: str):
    frags = generate_theoretical_fragments(annotate, mode)
    return tuple(frags)


def process_batch(args):
    annotate_batch, mode = args
    results = []
    for annotate in annotate_batch:
        result = cached_process_single(annotate, mode)
        results.append(list(result))
    return results


def parallel_process_with_cache(
    annotates,
    num_processes=4,
    batch_size=1000,
    prefer_threads=None,
    verbose=False,
    mode: str = "unmodified",
):
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
        iterator = pool.imap(process_batch, [(b, mode) for b in batches])
        if verbose:
            iterator = tqdm(iterator, total=len(batches), desc="theoretical fragments")
        for res in iterator:
            batch_results.append(res)

    final_results = []
    for batch_result in batch_results:
        final_results.extend(batch_result)

    return final_results


try:
    from numba import njit  # type: ignore

    _NUMBA = True
except ImportError:
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
    theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer, mode: str = "unmodified"
):
    """Match theoretical m/z to experimental peaks and take max intensity within tolerance.

    Returns a list like [[frag_str, matched_intensity], ...]. If mass_analyzer is
    not FTMS or ITMS, return None to mimic previous behavior (skip row).
    """
    mode = str(mode or "unmodified").strip().lower()
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

    names = []
    mz_vals = []
    by_mz_vals = []
    m_indices = []

    for idx, (frag_str, tmz) in enumerate(theory_mz_list):
        names.append(frag_str)
        mz_val = np.nan if tmz is None else float(tmz)
        mz_vals.append(mz_val)

        if mode != "phospho":
            continue
        if tmz is None or frag_str is None:
            continue

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

    is_ppm = mass_analyzer == "FTMS"
    tol_value = 20.0 if is_ppm else 0.5

    # phospho: if m-ion m/z overlaps with a b/y ion within tolerance, set it to NaN to avoid duplicate matching
    if mode == "phospho" and by_mz_vals and m_indices:
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
    theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer, mode = args
    return fast_intensity_matching(
        theory_mz_list, experiment_mz_list, experiment_int_list, mass_analyzer, mode=mode
    )


def parallel_intensity_matching(
    combined_df,
    num_processes=4,
    batch_size=1000,
    prefer_threads=None,
    verbose=False,
    mode: str = "unmodified",
):

    args_list = list(
        zip(
            combined_df["theoretical_fragments"],
            combined_df["mzarray"],
            combined_df["intarray"],
            combined_df["Mass analyzer"],
            repeat(mode),
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


def process_pair(meta_path, mz_path, msms_root, df_h5_dir, inner_num_procs=4, mode: str = "unmodified"):
    try:
        if oms is None:
            raise ImportError("Missing dependency: pyopenms (required for mzML reading)")
        file_name = os.path.basename(meta_path)
        MSMS = os.path.join(msms_root, file_name)

        df1 = pd.read_csv(MSMS, sep="\t", low_memory=False)

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
        meta_df = df1[columns_to_keep].copy()
        meta_df.columns = new_column_names

        meta_df["annotate"] = meta_df["Modified_sequence"].apply(apply_modifications)

        df1["Scan number"] = df1["Scan number"].astype(int)
        meta_df["MS2_Scan_Number"] = meta_df["MS2_Scan_Number"].astype(int)

        meta_df = meta_df.merge(
            df1[["Scan number", "Mass analyzer", "Fragmentation"]],
            left_on="MS2_Scan_Number",
            right_on="Scan number",
            how="left",
        )

        meta_df.drop(columns=["Scan number"], inplace=True)

        mode = str(mode or "unmodified").strip().lower()
        prefer_threads = True if mode == "phospho" else False

        meta_df["theoretical_fragments"] = parallel_process_with_cache(
            meta_df["annotate"].values.tolist(),
            num_processes=inner_num_procs,
            batch_size=500,
            prefer_threads=prefer_threads,
            verbose=False,
            mode=mode,
        )

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

                try:
                    short = [
                        x.decode() for x in precursor.getActivationMethodsAsShortString()
                    ]
                    frag = short[0] if short else ""
                except (AttributeError, TypeError, UnicodeDecodeError):
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
                    f"Fragmentation: mzML override={int(use_mzml.sum())}/{len(combined_df)}"
                )
            combined_df.drop(columns=["Fragmentation_mzml"], inplace=True)
        combined_df["collision_energy"] = (
            combined_df["collision_energy"].astype(float).fillna(30)
        )
        combined_df["theoretical_fragments_int"] = parallel_intensity_matching(
            combined_df,
            num_processes=inner_num_procs,
            batch_size=500,
            prefer_threads=prefer_threads,
            verbose=False,
            mode=mode,
        )
        output_stem = os.path.splitext(os.path.basename(meta_path))[0]
        output_path = os.path.join(df_h5_dir, output_stem + ".h5")
        _atomic_to_hdf(combined_df, output_path, key="combined_data")
        print(f"Processed and saved: {output_path}")
        gc.collect()
        return True
    except Exception as e:
        print(f"Error processing {meta_path} and {mz_path}: {str(e)}")
        return False


def run_step2(config):
    mode = str(config.get("mode", "unmodified")).strip().lower()
    msms_root = config["paths"]["msms_filtered_dir"]
    mzml_root = config["paths"]["mzml_dir"]
    search_root = config["paths"]["search_dir"]
    df_h5_dir = config["paths"]["df_h5_dir"]
    num_work = config["performance"]["num_workers"]

    os.makedirs(df_h5_dir, exist_ok=True)

    def _norm_stem(stem: str) -> str:
        s = str(stem).strip().lower()
        if s.endswith(".raw"):
            s = s[:-4]
        if s.endswith(".mzml"):
            s = s[:-5]
        return s

    mzml_by_norm = {}
    for fn in os.listdir(mzml_root):
        if not fn.lower().endswith(".mzml"):
            continue
        stem = os.path.splitext(fn)[0]
        key = _norm_stem(stem)
        mzml_by_norm.setdefault(key, []).append(os.path.join(mzml_root, fn))

    search_entries = sorted(fn for fn in os.listdir(search_root) if fn.endswith(".txt"))

    valid_mzml_files = []
    valid_search_files = []

    for fn in search_entries:
        stem = os.path.splitext(fn)[0]
        key = _norm_stem(stem)
        candidates = mzml_by_norm.get(key, [])
        search_path = os.path.join(search_root, fn)
        if len(candidates) == 1:
            valid_search_files.append(search_path)
            valid_mzml_files.append(candidates[0])
        elif len(candidates) == 0:
            if mode == "phospho":
                fallback = [
                    os.path.join(mzml_root, f)
                    for f in os.listdir(mzml_root)
                    if f.lower().endswith(".mzml") and stem in os.path.splitext(f)[0]
                ]
                if fallback:
                    fallback.sort()
                    print(
                        f"Warning: phospho mode - no exact mzML match, using substring: {stem} -> {os.path.basename(fallback[0])}"
                    )
                    valid_search_files.append(search_path)
                    valid_mzml_files.append(fallback[0])
                    continue
            print(f"Warning: No mzML file found for {stem}")
        else:
            print(f"Warning: Multiple mzML files found for {stem}:")
            for p in candidates:
                print(f"  - {p}")

    print(f"Found {len(valid_search_files)} valid file pairs to process")

    force = str(os.environ.get("FORCE", "0")).strip() == "1"
    inner_num_procs = num_work

    results = []
    for search_file, mzml_file in tqdm(
        zip(valid_search_files, valid_mzml_files),
        total=len(valid_search_files),
        desc="Processing files",
    ):
        stem = os.path.splitext(os.path.basename(search_file))[0]
        expected_out = os.path.join(df_h5_dir, stem + ".h5")
        if (not force) and _is_valid_combined_h5(expected_out):
            print(f"[SKIP] 已存在：{expected_out}")
            results.append(True)
            continue

        result = process_pair(
            search_file,
            mzml_file,
            msms_root,
            df_h5_dir,
            inner_num_procs,
            mode=mode,
        )
        results.append(result)
        gc.collect()

    successful = sum(1 for r in results if r)
    print(f"Successfully processed {successful} out of {len(results)} file pairs")
    print("3.2完成")


if __name__ == "__main__":
    import os

    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step2(config)
