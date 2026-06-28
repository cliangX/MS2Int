#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build candidate-level reference spectrum H5 from DeepFLR target_decoy CSV.

Input: step1_target_decoy.csv + msms.txt + mzML directory
Output: H5 with per-candidate train_data (N, 39, 41) aligned with Mamba predictions.

Reuses phosphorylation.step2/step3 fragment generation and intensity matching.
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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
EXTRACT_ROOT = os.path.join(CURRENT_DIR, "phosphorylation")

if EXTRACT_ROOT not in sys.path:
    sys.path.insert(0, EXTRACT_ROOT)

from step2_process_df_h5 import (  # type: ignore
    cached_process_single,
    fast_intensity_matching,
)
from step3_generate_train_data import (  # type: ignore
    H5_SPECTRUM_SHAPE,
    ION_ROWS,
    ION_COLS,
    ION_TO_IDX,
    ANNOTATION_MATRIX,
)

SPECTRUM_UTILS_AVAILABLE = False
proforma = None  # type: ignore
fragment_annotation = None  # type: ignore
import importlib
if importlib.util.find_spec("spectrum_utils") is not None:
    from spectrum_utils import proforma, fragment_annotation  # type: ignore
    SPECTRUM_UTILS_AVAILABLE = True


def _normalize_ion_name(name: str):
    if not name:
        return None

    base = str(name)
    charge_suffix = ""
    if "-H3PO4" in base:
        base = base.replace("-H3PO4", "")

    if "^" in base:
        parts = base.split("^", 1)
        base, charge_suffix = parts[0], "^" + parts[1]

    if not base:
        return None

    if base.startswith("m"):
        return base

    return base + charge_suffix


def compute_theoretical_mz_grid(annotate: str, charge: int) -> np.ndarray:
    if not SPECTRUM_UTILS_AVAILABLE:
        raise ImportError("spectrum_utils required for theoretical m/z computation")
    if ANNOTATION_MATRIX is None:
        raise ImportError("ANNOTATION_MATRIX not available")
    
    seq = proforma.parse(annotate)
    if not seq:
        return np.full(H5_SPECTRUM_SHAPE, np.nan, dtype=np.float32)
    seq = seq[0]

    theoretical_fragments = fragment_annotation.get_theoretical_fragments(
        seq,
        ion_types="byIm",
        max_charge=2,
        neutral_losses={"H3PO4": -97.976896},
    )
    
    ion_mz_map = {}
    for ion_name, mz_value in theoretical_fragments:
        if mz_value is not None:
            ion_mz_map[str(ion_name)] = float(mz_value)
    
    mz_grid = np.full(H5_SPECTRUM_SHAPE, np.nan, dtype=np.float32)
    for row in range(ION_COLS):
        for col in range(ION_ROWS):
            ion_name = ANNOTATION_MATRIX[col, row]
            if ion_name and ion_name in ion_mz_map:
                mz_grid[row, col] = ion_mz_map[ion_name]
    
    return mz_grid


def generate_by_priority_mask(mz_grid: np.ndarray) -> np.ndarray:
    if ANNOTATION_MATRIX is None:
        raise ImportError("ANNOTATION_MATRIX not available")

    L, V = mz_grid.shape
    mask = np.ones((L, V), dtype=np.uint8)

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
    if not SPECTRUM_UTILS_AVAILABLE:
        return np.ones(H5_SPECTRUM_SHAPE, dtype=np.uint8)
    
    mz_grid = compute_theoretical_mz_grid(annotate, charge)
    return generate_by_priority_mask(mz_grid)


def convert_deepflr_to_mamba_sequence(key_x: str) -> str:
    modification_map = {
        "1": "[Phospho]",
        "2": "[Oxidation]",
        "3": "[Carbamidomethyl]",
        "4": "[Acetyl]",
    }

    mamba_sequence = key_x

    if mamba_sequence.startswith("4"):
        mamba_sequence = "[Acetyl]-" + mamba_sequence[1:]

    for deepflr_code, mamba_mod in modification_map.items():
        if deepflr_code == "4":
            continue
        pattern = f"([A-Z]){deepflr_code}"
        replacement = f"\\1{mamba_mod}"
        mamba_sequence = re.sub(pattern, replacement, mamba_sequence)

    return mamba_sequence


def load_msms_msinfo(
    msms_path: str,
) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], str]]:
    df = pd.read_csv(msms_path, sep="\t", low_memory=False)

    required_cols = ["Raw file", "Scan number", "Mass analyzer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"msms.txt missing required column: {col}")

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
    if not os.path.isfile(mzml_path):
        raise FileNotFoundError(f"mzML file not found: {mzml_path}")

    exp = oms.MSExperiment()
    oms.MzMLFile().load(mzml_path, exp)

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
        precursors = spectrum.getPrecursors()
        if precursors:
            precursor = precursors[0]
            if precursor.metaValueExists("collision energy"):
                ce = float(precursor.getMetaValue("collision energy"))

        scan_number = idx + 1
        ms2_map[scan_number] = (mz, inten, rt, ce)

    return ms2_map


def build_train_matrix(frag_int_pairs: List[Tuple[str, float]]) -> np.ndarray:
    flat_len = ION_ROWS * ION_COLS
    vec = np.zeros(flat_len, dtype=float)

    for name, inten in frag_int_pairs:
        norm_name = _normalize_ion_name(name)
        if norm_name is None:
            continue
        
        idx = ION_TO_IDX.get(norm_name)
        if idx is None:
            continue
        
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
    zero_train = np.zeros((ION_ROWS, ION_COLS), dtype=float)
    default_analyzer = "FTMS"
    default_frag = ""
    default_ce = float("nan")

    annotate = convert_deepflr_to_mamba_sequence(key_seq)

    theory_list = cached_process_single(annotate)
    if not theory_list:
        return zero_train, float("nan"), default_analyzer, default_frag, default_ce

    mz_int_rt_ce = ms2_map.get(scan)
    if mz_int_rt_ce is None:
        return zero_train, float("nan"), default_analyzer, default_frag, default_ce

    mz_array, inten_array, rt, ce = mz_int_rt_ce

    analyzer = msinfo.get((raw, scan), default_analyzer)
    frag = fraginfo.get((raw, scan), default_frag)

    matched = fast_intensity_matching(
        theory_list,
        mz_array,
        inten_array,
        analyzer,
    )

    if matched is None:
        return zero_train, rt, analyzer, frag, ce

    ion_int_pairs = [(str(n), float(v)) for n, v in matched]
    train_mat = build_train_matrix(ion_int_pairs)
    return train_mat, rt, analyzer, frag, ce


def _process_one_raw_worker(args: Tuple[str, List[int], str, Dict[Tuple[str, int], str], Dict[Tuple[str, int], str], pd.DataFrame]) -> Dict[str, Any]:
    raw, idx_list, mzml_dir, msinfo, fraginfo, td_subset = args
    
    results: Dict[int, Tuple[np.ndarray, np.ndarray, float, str, str, float]] = {}

    mzml_path = os.path.join(mzml_dir, f"{raw}.mzML")
    if os.path.isfile(mzml_path):
        ms2_map = load_mzml_ms2_map(mzml_path)
    else:
        ms2_map = {}

    for _, row in td_subset.iterrows():
        idx = int(row["_idx"])
        scan = row["Spectrum"]
        key_seq = row["key"]
        charge = row["PP.Charge"]
        
        if pd.isna(scan) or not key_seq:
            train_mat = np.zeros((ION_ROWS, ION_COLS), dtype=float)
            by_mask = np.ones(H5_SPECTRUM_SHAPE, dtype=np.uint8)
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

        annotate = convert_deepflr_to_mamba_sequence(str(key_seq))
        charge_val = int(charge) if not pd.isna(charge) else 2
        by_mask = compute_by_priority_mask_for_sequence(annotate, charge_val)
        
        results[idx] = (train_mat, by_mask, rt, analyzer, frag, ce)
    
    return {"raw": raw, "results": results}


def build_ref_h5(
    target_decoy_csv: str,
    msms_path: str,
    mzml_dir: str,
    output_h5: str,
    quiet: bool = False,
    num_workers: int = 1,
) -> None:
    if not os.path.isfile(target_decoy_csv):
        raise FileNotFoundError(f"target_decoy file not found: {target_decoy_csv}")
    if not os.path.isfile(msms_path):
        raise FileNotFoundError(f"msms.txt not found: {msms_path}")
    if not os.path.isdir(mzml_dir):
        raise NotADirectoryError(f"mzML directory not found: {mzml_dir}")

    td = pd.read_csv(target_decoy_csv)

    required_cols = ["SourceFile", "Spectrum", "PP.Charge", "key", "exp_strip_sequence"]
    for col in required_cols:
        if col not in td.columns:
            raise ValueError(f"target_decoy file missing required column: {col}")

    td["SourceFile"] = td["SourceFile"].astype(str)
    td["Spectrum"] = pd.to_numeric(td["Spectrum"], errors="coerce").astype("Int64")
    td["PP.Charge"] = pd.to_numeric(td["PP.Charge"], errors="coerce").astype("Int64")
    td["key"] = td["key"].astype(str)
    td["exp_strip_sequence"] = td["exp_strip_sequence"].astype(str)

    msinfo, fraginfo = load_msms_msinfo(msms_path)

    n_rows = len(td)
    print(f"target_decoy candidates: {n_rows}")

    train_mats: List[np.ndarray] = [None] * n_rows  # type: ignore
    by_priority_masks: List[np.ndarray] = [None] * n_rows  # type: ignore
    rt_list: List[float] = [float("nan")] * n_rows
    analyzer_list: List[str] = ["" for _ in range(n_rows)]
    frag_list: List[str] = ["" for _ in range(n_rows)]
    ce_list: List[float] = [float("nan")] * n_rows

    grouped_indices: Dict[str, List[int]] = {}
    for idx, raw in enumerate(td["SourceFile"]):
        if pd.isna(raw):
            continue
        grouped_indices.setdefault(str(raw), []).append(idx)

    td["_idx"] = td.index

    task_args_list = []
    for raw, idx_list in grouped_indices.items():
        td_subset = td.loc[idx_list, ["_idx", "Spectrum", "key", "PP.Charge"]].copy()
        task_args_list.append((raw, idx_list, mzml_dir, msinfo, fraginfo, td_subset))

    if num_workers <= 1:
        iterator = tqdm(task_args_list, desc="Raw files", unit="raw", disable=quiet)
        for args in iterator:
            result = _process_one_raw_worker(args)
            for idx, (train_mat, by_mask, rt, analyzer, frag, ce) in result["results"].items():
                train_mats[idx] = train_mat
                by_priority_masks[idx] = by_mask
                rt_list[idx] = float(rt) if not np.isnan(rt) else float("nan")
                analyzer_list[idx] = str(analyzer)
                frag_list[idx] = str(frag)
                ce_list[idx] = float(ce) if not np.isnan(ce) else float("nan")
            gc.collect()
    else:
        with mp.Pool(processes=num_workers) as pool:
            results_iter = tqdm(
                pool.imap_unordered(_process_one_raw_worker, task_args_list),
                total=len(task_args_list),
                desc="Raw files", unit="raw", disable=quiet,
            )
            for result in results_iter:
                for idx, (train_mat, by_mask, rt, analyzer, frag, ce) in result["results"].items():
                    train_mats[idx] = train_mat
                    by_priority_masks[idx] = by_mask
                    rt_list[idx] = float(rt) if not np.isnan(rt) else float("nan")
                    analyzer_list[idx] = str(analyzer)
                    frag_list[idx] = str(frag)
                    ce_list[idx] = float(ce) if not np.isnan(ce) else float("nan")
        
        gc.collect()

    for i in range(n_rows):
        if train_mats[i] is None:
            train_mats[i] = np.zeros((ION_ROWS, ION_COLS), dtype=float)
        if by_priority_masks[i] is None:
            by_priority_masks[i] = np.ones(H5_SPECTRUM_SHAPE, dtype=np.uint8)

    train_array = np.stack(train_mats, axis=0)
    train_array = np.swapaxes(train_array, 1, 2)  # (N, ION_ROWS, ION_COLS) -> (N, 39, 41)

    by_priority_mask_array = np.stack(by_priority_masks, axis=0)

    analyzers = np.asarray(analyzer_list, dtype="S32")
    frags = np.asarray(frag_list, dtype="S16")
    collision_energies = np.array(ce_list, dtype=np.float32)

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    raw_files = td["SourceFile"].astype(str).to_numpy(dtype="S100")
    scans = td["Spectrum"].astype("int64").to_numpy()
    charges = td["PP.Charge"].astype("int64").to_numpy()
    sequences = td["exp_strip_sequence"].astype(str).to_numpy(dtype="S128")
    keys = td["key"].astype(str).to_numpy(dtype="S256")
    rts = np.array(rt_list, dtype=np.float32)

    with h5py.File(output_h5, "w") as f:
        dset = f.create_dataset("Raw_file", data=raw_files)
        dset.attrs["description"] = "Raw file name"

        dset = f.create_dataset("MS2_Scan_Number", data=scans)
        dset.attrs["description"] = "MS2 scan number"

        dset = f.create_dataset("Charge", data=charges)
        dset.attrs["description"] = "Charge state"

        dset = f.create_dataset("Sequence", data=sequences)
        dset.attrs["description"] = "Stripped amino acid sequence"

        dset = f.create_dataset("key", data=keys)
        dset.attrs["description"] = "DeepFLR encoded sequence"

        dset = f.create_dataset("RT", data=rts)
        dset.attrs["description"] = "Retention time"

        dset = f.create_dataset("Mass_analyzer", data=analyzers)
        dset.attrs["description"] = "Mass analyzer type"

        dset = f.create_dataset("Fragmentation", data=frags)
        dset.attrs["description"] = "Fragmentation method"

        dset = f.create_dataset("collision_energy", data=collision_energies)
        dset.attrs["description"] = "Collision energy"

        dset = f.create_dataset("train_data", data=train_array)
        dset.attrs["description"] = "Intensity matrix (N, 39, 41)"

        dset = f.create_dataset("by_priority_mask", data=by_priority_mask_array)
        dset.attrs["description"] = "b/y priority mask (N, 39, 41)"

        f.attrs["description"] = "Candidate-level reference spectrum H5"
        f.attrs["source_target_decoy"] = os.path.abspath(target_decoy_csv)
        f.attrs["source_msms"] = os.path.abspath(msms_path)
        f.attrs["mzml_dir"] = os.path.abspath(mzml_dir)

    print(f"train_data shape: {train_array.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-level reference spectrum H5 from DeepFLR target_decoy CSV + msms.txt + mzML"
    )
    parser.add_argument(
        "--target_decoy_csv",
        required=True,
        help="Path to step1_target_decoy.csv",
    )
    parser.add_argument(
        "--msms",
        required=True,
        help="Path to MaxQuant msms.txt",
    )
    parser.add_argument(
        "--mzml-dir",
        required=True,
        help="Directory containing mzML files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output H5 file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (disable tqdm and most messages)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32; set to 1 for serial)",
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
