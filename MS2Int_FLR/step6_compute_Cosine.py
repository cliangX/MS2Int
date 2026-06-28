#!/usr/bin/env python3
"""Compute spectral similarity (cosine) between predicted and reference spectra."""

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

import importlib
SPECTRUM_UTILS_AVAILABLE = importlib.util.find_spec("spectrum_utils") is not None
if SPECTRUM_UTILS_AVAILABLE:
    from spectrum_utils import fragment_annotation, proforma
else:
    fragment_annotation = None
    proforma = None


def _to_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_ROOT = os.path.join(SCRIPT_DIR, "phosphorylation")
if EXTRACT_ROOT not in sys.path:
    sys.path.insert(0, EXTRACT_ROOT)

ANNOTATION_MATRIX = None  # type: ignore
ION_ROWS = None  # type: ignore
ION_COLS = None  # type: ignore
H5_SPECTRUM_SHAPE = None  # type: ignore
if importlib.util.find_spec("step3_generate_train_data") is not None or os.path.isfile(
    os.path.join(EXTRACT_ROOT, "step3_generate_train_data.py")
):
    from step3_generate_train_data import (  # type: ignore
        ANNOTATION_MATRIX,
        H5_SPECTRUM_SHAPE,
        ION_COLS,
        ION_ROWS,
    )


def compute_theoretical_mz_grid(annotate: str, charge: int) -> np.ndarray:
    """Compute theoretical m/z grid (39, 41) for a peptide sequence."""
    if proforma is None or fragment_annotation is None:
        raise ImportError("spectrum_utils required")
    if ANNOTATION_MATRIX is None or H5_SPECTRUM_SHAPE is None:
        raise ImportError("ANNOTATION_MATRIX not available")

    seq = proforma.parse(annotate)
    if not seq:
        return np.full(H5_SPECTRUM_SHAPE, np.nan, dtype=np.float32)
    seq = seq[0]

    theoretical_fragments = fragment_annotation.get_theoretical_fragments(
        seq, ion_types="byIm", max_charge=2,
        neutral_losses={"H3PO4": -97.976896},
    )

    ion_mz_map = {}
    for fragment, mz_value in theoretical_fragments:
        if mz_value is not None:
            ion_mz_map[str(fragment)] = float(mz_value)

    mz_grid = np.full(H5_SPECTRUM_SHAPE, np.nan, dtype=np.float32)
    for row in range(ION_COLS):
        for col in range(ION_ROWS):
            ion_name = ANNOTATION_MATRIX[col, row]
            if ion_name and ion_name in ion_mz_map:
                mz_grid[row, col] = ion_mz_map[ion_name]

    return mz_grid


def generate_by_priority_mask(mz_grid: np.ndarray, mass_analyzer: str = "FTMS") -> np.ndarray:
    """Generate b/y priority mask: zero out m-ions that conflict with b/y m/z."""
    if ANNOTATION_MATRIX is None:
        raise ImportError("ANNOTATION_MATRIX not available")

    L, V = mz_grid.shape
    mask = np.ones((L, V), dtype=np.uint8)

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
    """Apply b/y priority mask to a batch of predicted spectra."""
    N, L, V = spectra.shape
    result = spectra.copy() if copy_input else spectra

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
        mz_grid = compute_theoretical_mz_grid(seq, charge)[:, :V]
        mask = generate_by_priority_mask(mz_grid)
        return key, mask

    if num_workers > 1 and len(unique_keys) > 1:
        max_workers = min(int(num_workers), len(unique_keys))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for key, mask in ex.map(_compute_mask_for_key, unique_keys):
                masks[key] = mask
    else:
        for key in unique_keys:
            k, mask = _compute_mask_for_key(key)
            masks[k] = mask

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
    parser.add_argument("--pred_h5", required=True)
    parser.add_argument("--ref_h5", required=True)
    parser.add_argument("--pred_key", default="Intpredict")
    parser.add_argument("--true_key", default="train_data")
    parser.add_argument("--n", required=True, nargs="+")
    parser.add_argument("--mode", choices=["flatten", "per-position"], default="flatten")
    parser.add_argument("--align", choices=["scan", "index"], default="scan")
    parser.add_argument("--template_csv", default="script/test_modelresult_template.csv")
    parser.add_argument("--by_priority_workers", type=int, default=max(1, min(120, os.cpu_count() or 1)))
    return parser.parse_args()


def _split_n_tokens(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    token = token.replace(",", " ")
    return [piece for piece in token.split() if piece]


def parse_n_values(raw_values: List[str]) -> List[int]:
    if not raw_values:
        raise ValueError("--n requires at least one value.")

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
        raise ValueError("--n parsed to empty list.")

    result: List[int] = []
    seen = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"--n requires positive integers, got: {value}")
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def as_str_array(ds) -> np.ndarray:
    """Decode an HDF5 bytes dataset to numpy array of str."""
    if hasattr(ds, "asstr"):
        return ds.asstr()[:]
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
        pred_sequences: (N_pred,) str
        pred_charges: (N_pred,) int
        pred_analyzers: (N_pred,) str
        by_priority_mask: (N_ref, L, V) uint8 or None
    """
    with h5py.File(pred_h5, "r") as fp:
        if pred_key not in fp:
            raise KeyError(f"Dataset '{pred_key}' not found in {pred_h5}")
        if "Raw_file" not in fp or "MS2_Scan_Number" not in fp:
            raise KeyError("pred H5 missing Raw_file or MS2_Scan_Number")
        y_pred_full = fp[pred_key][:].astype(np.float32, copy=False)
        pred_raw = as_str_array(fp["Raw_file"])
        pred_scan = fp["MS2_Scan_Number"][:].astype(np.int64, copy=False)
        
        pred_sequences = as_str_array(fp["Sequence"]) if "Sequence" in fp else None
        pred_charges = fp["Charge"][:].astype(np.int64, copy=False) if "Charge" in fp else None
        pred_analyzers = as_str_array(fp["Mass_analyzer"]) if "Mass_analyzer" in fp else None

    with h5py.File(ref_h5, "r") as fr:
        if true_key not in fr:
            raise KeyError(f"Dataset '{true_key}' not found in {ref_h5}")
        if "Raw_file" not in fr or "MS2_Scan_Number" not in fr:
            raise KeyError("ref H5 missing Raw_file or MS2_Scan_Number")
        y_true_full = fr[true_key][:].astype(np.float32, copy=False)
        ref_raw = as_str_array(fr["Raw_file"])
        ref_scan = fr["MS2_Scan_Number"][:].astype(np.int64, copy=False)
        
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
    """Flatten (L,V) -> 1D, mask y_true!=-1, sqrt+L2 cosine similarity."""
    b, l, v = y_true.shape
    y_true_flat = y_true.reshape(b, -1)
    y_pred_flat = y_pred.reshape(b, -1)
    mask = (y_true_flat != -1).float()
    y_true_masked = y_true_flat * mask
    y_pred_masked = y_pred_flat * mask
    y_true_norm = F.normalize(torch.sqrt(y_true_masked), p=2, dim=-1)
    y_pred_norm = F.normalize(torch.sqrt(y_pred_masked), p=2, dim=-1)
    return torch.sum(y_true_norm * y_pred_norm, dim=-1).clamp(-1.0, 1.0)


def masked_spectral_distance_perpos(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-position sqrt+L2 cosine similarity, averaged over length."""
    mask = (y_true != -1).float()
    y_true_m = y_true * mask
    y_pred_m = y_pred * mask
    y_true_norm = F.normalize(torch.sqrt(y_true_m), p=2, dim=-1)
    y_pred_norm = F.normalize(torch.sqrt(y_pred_m), p=2, dim=-1)
    cos_sim = torch.sum(y_true_norm * y_pred_norm, dim=-1).clamp(-1.0, 1.0)
    return cos_sim.mean(dim=-1)


def compute_scores(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    mode: str,
    extra_mask_np: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spectral similarity scores."""
    y_true = torch.tensor(y_true_np, dtype=torch.float32)
    y_pred = torch.tensor(y_pred_np, dtype=torch.float32)

    if extra_mask_np is not None:
        if extra_mask_np.shape != y_true_np.shape:
            raise ValueError(
                f"extra_mask shape {extra_mask_np.shape} != y_true shape {y_true_np.shape}"
            )
        mask_bool = torch.tensor(extra_mask_np != 0, dtype=torch.bool)
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
        print(f"Written dataset: {name}")


def update_template_csv(template_csv: str, scores_full: np.ndarray) -> None:
    df = pd.read_csv(template_csv)
    if len(df) != len(scores_full):
        raise ValueError(
            f"CSV rows {len(df)} != prediction samples {len(scores_full)}"
        )
    df["score"] = scores_full
    df.to_csv(template_csv, index=False)


def main() -> None:
    args = parse_args()
    n_values = parse_n_values(args.n)

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
            f"Length mismatch: pred L={y_pred_full.shape[1]} vs true L={y_true_full.shape[1]}"
        )

    if args.align == "scan":
        matched_pred_idx, matched_ref_idx, missing = build_alignment(
            pred_raw, pred_scan, ref_raw, ref_scan
        )
        N_pred = y_pred_full.shape[0]
    else:
        N_pred = y_pred_full.shape[0]
        N_ref = y_true_full.shape[0]
        if N_pred != N_ref:
            raise ValueError(
                f"Index align requires equal sizes: pred N={N_pred} vs true N={N_ref}"
            )
        matched_pred_idx = np.arange(N_pred, dtype=np.int64)
        matched_ref_idx = np.arange(N_ref, dtype=np.int64)
        missing = 0

    multi = len(n_values) > 1
    max_pred_channels = y_pred_full.shape[2]
    max_true_channels = y_true_full.shape[2]

    max_n = max(n_values)
    if max_n > max_pred_channels or max_n > max_true_channels:
        raise ValueError(
            f"max(n)={max_n} exceeds channel dim: pred={max_pred_channels}, true={max_true_channels}"
        )

    y_pred_matched_max = y_pred_full[:, :, :max_n][matched_pred_idx]

    if multi:
        y_true_matched_max = y_true_full[:, :, :max_n][matched_ref_idx]
    else:
        y_true_matched_max = None

    del y_pred_full

    if by_priority_mask_full is not None:
        mask_matched = by_priority_mask_full[matched_ref_idx][:, :, :max_n]
        y_pred_matched_max *= mask_matched.astype(np.float32)
        if multi and y_true_matched_max is not None:
            y_true_matched_max *= mask_matched.astype(np.float32)
    elif pred_sequences is not None and pred_charges is not None and pred_analyzers is not None:
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
        pass

    for idx, n in enumerate(n_values, start=1):
        if n > max_pred_channels or n > max_true_channels:
            raise ValueError(
                f"n={n} exceeds channel dim: pred={max_pred_channels}, true={max_true_channels}"
            )

        y_pred_crop = y_pred_matched_max[:, :, :n]
        if multi:
            assert y_true_matched_max is not None
            y_true_crop = y_true_matched_max[:, :, :n]
        else:
            y_true_crop = y_true_full[:, :, :n][matched_ref_idx].copy()
            if by_priority_mask_full is not None:
                mask_crop = by_priority_mask_full[matched_ref_idx][:, :, :n]
                y_true_crop *= mask_crop.astype(np.float32)

        scores_part = compute_scores(
            y_true_crop,
            y_pred_crop,
            args.mode,
            extra_mask_np=None,
        )

        scores_full = np.full((N_pred,), np.nan, dtype=np.float32)
        scores_full[matched_pred_idx] = scores_part

        ds_name = "Cosine_Similarity"
        write_dataset(args.pred_h5, ds_name, scores_full)

        update_template_csv(args.template_csv, scores_full)


if __name__ == "__main__":
    main()
