#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MS2PIP feature calculator with m-ion support (quiet mode)

This script explicitly suppresses TensorFlow/absl GPU init warnings that may
appear in some environments when spawned in multiple worker processes. We do
not use TensorFlow here, so disable GPU and lower TF log level early.
"""

# Suppress TensorFlow/absl GPU init warnings before any possible import
import os as _os  # noqa: E402

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=INFO,2=WARNING,3=ERROR
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # disable GPU visibility for TF
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoid extra oneDNN logs

import argparse
import logging as _logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import importlib.util as _ilu
if _ilu.find_spec("add_specid_to_h5"):
    from add_specid_to_h5 import add_specid_to_h5 as _add_specid_to_h5
else:
    _add_specid_to_h5 = None

_logging.getLogger("numba").setLevel(_logging.WARNING)

NUMBA_AVAILABLE = _ilu.find_spec("numba") is not None
if NUMBA_AVAILABLE:
    from numba import njit

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _mse_numba(x, y):
        return np.mean((x - y) ** 2)

    @njit(cache=True, fastmath=True)
    def _cos_numba(x, y):
        nx = np.sqrt((x * x).sum())
        ny = np.sqrt((y * y).sum())
        return 0.0 if nx == 0.0 or ny == 0.0 else (x * y).sum() / (nx * ny)
else:

    def _mse_numba(x, y):
        return float(np.mean((x - y) ** 2))

    def _cos_numba(x, y):
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        return 0.0 if nx == 0.0 or ny == 0.0 else float(np.dot(x, y) / (nx * ny))


def calculate_ms2pip_features(intpredict, train_data, length):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        targ_b = np.concatenate([intpredict[:length, 0], intpredict[:length, 1]])
        targ_y = np.concatenate([intpredict[:length, 2], intpredict[:length, 3]])
        targ_m = np.concatenate(
            [intpredict[:length, 5], intpredict[:length, 6], intpredict[:length, 7]]
        )
        targ_all = np.concatenate([targ_b, targ_y])

        pred_b = np.concatenate([train_data[:length, 0], train_data[:length, 1]])
        pred_y = np.concatenate([train_data[:length, 2], train_data[:length, 3]])
        pred_m = np.concatenate(
            [train_data[:length, 5], train_data[:length, 6], train_data[:length, 7]]
        )
        pred_all = np.concatenate([pred_b, pred_y])

        max_pred = float(np.max(pred_all))
        if max_pred > 0.0:
            pred_b = pred_b / max_pred
            pred_y = pred_y / max_pred
            pred_m = pred_m / max_pred
            pred_all = pred_all / max_pred

        def _spearman(x, y):
            if len(x) < 2:
                return 0.0
            r = np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
            return 0.0 if np.isnan(r) else float(r)

        eps = 0.001
        log = lambda x: np.log2(x + eps).clip(np.log2(eps))

        tb_l, ty_l, ta_l = log(targ_b), log(targ_y), log(targ_all)
        pb_l, py_l, pa_l = log(pred_b), log(pred_y), log(pred_all)
        tm_l = log(targ_m)
        pm_l = log(pred_m)

        adb, ady, ada = (
            np.abs(targ_b - pred_b),
            np.abs(targ_y - pred_y),
            np.abs(targ_all - pred_all),
        )
        adb_l, ady_l, ada_l = (
            np.abs(tb_l - pb_l),
            np.abs(ty_l - py_l),
            np.abs(ta_l - pa_l),
        )
        adm = np.abs(targ_m - pred_m)
        adm_l = np.abs(tm_l - pm_l)

        features = {
            "spec_pearson_norm": np.corrcoef(ta_l, pa_l)[0, 1],
            "ionb_pearson_norm": np.corrcoef(tb_l, pb_l)[0, 1],
            "iony_pearson_norm": np.corrcoef(ty_l, py_l)[0, 1],
            "spec_mse_norm": _mse_numba(ta_l, pa_l),
            "ionb_mse_norm": _mse_numba(tb_l, pb_l),
            "iony_mse_norm": _mse_numba(ty_l, py_l),
            "min_abs_diff_norm": np.min(ada_l),
            "max_abs_diff_norm": np.max(ada_l),
            "abs_diff_Q1_norm": np.quantile(ada_l, 0.25),
            "abs_diff_Q2_norm": np.quantile(ada_l, 0.5),
            "abs_diff_Q3_norm": np.quantile(ada_l, 0.75),
            "mean_abs_diff_norm": np.mean(ada_l),
            "std_abs_diff_norm": np.std(ada_l),
            "ionb_min_abs_diff_norm": np.min(adb_l),
            "ionb_max_abs_diff_norm": np.max(adb_l),
            "ionb_abs_diff_Q1_norm": np.quantile(adb_l, 0.25),
            "ionb_abs_diff_Q2_norm": np.quantile(adb_l, 0.5),
            "ionb_abs_diff_Q3_norm": np.quantile(adb_l, 0.75),
            "ionb_mean_abs_diff_norm": np.mean(adb_l),
            "ionb_std_abs_diff_norm": np.std(adb_l),
            "iony_min_abs_diff_norm": np.min(ady_l),
            "iony_max_abs_diff_norm": np.max(ady_l),
            "iony_abs_diff_Q1_norm": np.quantile(ady_l, 0.25),
            "iony_abs_diff_Q2_norm": np.quantile(ady_l, 0.5),
            "iony_abs_diff_Q3_norm": np.quantile(ady_l, 0.75),
            "iony_mean_abs_diff_norm": np.mean(ady_l),
            "iony_std_abs_diff_norm": np.std(ady_l),
            "dotprod_norm": np.dot(ta_l, pa_l),
            "dotprod_ionb_norm": np.dot(tb_l, pb_l),
            "dotprod_iony_norm": np.dot(ty_l, py_l),
            "cos_norm": _cos_numba(ta_l, pa_l),
            "cos_ionb_norm": _cos_numba(tb_l, pb_l),
            "cos_iony_norm": _cos_numba(ty_l, py_l),
            "ionm_pearson_norm": np.corrcoef(tm_l, pm_l)[0, 1],
            "ionm_mse_norm": _mse_numba(tm_l, pm_l),
            "ionm_min_abs_diff_norm": np.min(adm_l),
            "ionm_max_abs_diff_norm": np.max(adm_l),
            "ionm_abs_diff_Q1_norm": np.quantile(adm_l, 0.25),
            "ionm_abs_diff_Q2_norm": np.quantile(adm_l, 0.5),
            "ionm_abs_diff_Q3_norm": np.quantile(adm_l, 0.75),
            "ionm_mean_abs_diff_norm": np.mean(adm_l),
            "ionm_std_abs_diff_norm": np.std(adm_l),
            "ionm_dotprod_norm": np.dot(tm_l, pm_l),
            "ionm_cos_norm": _cos_numba(tm_l, pm_l),
            "spec_pearson": np.corrcoef(targ_all, pred_all)[0, 1],
            "ionb_pearson": np.corrcoef(targ_b, pred_b)[0, 1],
            "iony_pearson": np.corrcoef(targ_y, pred_y)[0, 1],
            "spec_spearman": _spearman(targ_all, pred_all),
            "ionb_spearman": _spearman(targ_b, pred_b),
            "iony_spearman": _spearman(targ_y, pred_y),
            "spec_mse": _mse_numba(targ_all, pred_all),
            "ionb_mse": _mse_numba(targ_b, pred_b),
            "iony_mse": _mse_numba(targ_y, pred_y),
            "min_abs_diff_iontype": 0 if np.min(adb) <= np.min(ady) else 1,
            "max_abs_diff_iontype": 0 if np.max(adb) >= np.max(ady) else 1,
            "min_abs_diff": np.min(ada),
            "max_abs_diff": np.max(ada),
            "abs_diff_Q1": np.quantile(ada, 0.25),
            "abs_diff_Q2": np.quantile(ada, 0.5),
            "abs_diff_Q3": np.quantile(ada, 0.75),
            "mean_abs_diff": np.mean(ada),
            "std_abs_diff": np.std(ada),
            "ionb_min_abs_diff": np.min(adb),
            "ionb_max_abs_diff": np.max(adb),
            "ionb_abs_diff_Q1": np.quantile(adb, 0.25),
            "ionb_abs_diff_Q2": np.quantile(adb, 0.5),
            "ionb_abs_diff_Q3": np.quantile(adb, 0.75),
            "ionb_mean_abs_diff": np.mean(adb),
            "ionb_std_abs_diff": np.std(adb),
            "iony_min_abs_diff": np.min(ady),
            "iony_max_abs_diff": np.max(ady),
            "iony_abs_diff_Q1": np.quantile(ady, 0.25),
            "iony_abs_diff_Q2": np.quantile(ady, 0.5),
            "iony_abs_diff_Q3": np.quantile(ady, 0.75),
            "iony_mean_abs_diff": np.mean(ady),
            "iony_std_abs_diff": np.std(ady),
            "dotprod": np.dot(targ_all, pred_all),
            "dotprod_ionb": np.dot(targ_b, pred_b),
            "dotprod_iony": np.dot(targ_y, pred_y),
            "cos": _cos_numba(targ_all, pred_all),
            "cos_ionb": _cos_numba(targ_b, pred_b),
            "cos_iony": _cos_numba(targ_y, pred_y),
            "ionm_pearson": np.corrcoef(targ_m, pred_m)[0, 1],
            "ionm_spearman": _spearman(targ_m, pred_m),
            "ionm_mse": _mse_numba(targ_m, pred_m),
            "ionm_min_abs_diff": np.min(adm),
            "ionm_max_abs_diff": np.max(adm),
            "ionm_abs_diff_Q1": np.quantile(adm, 0.25),
            "ionm_abs_diff_Q2": np.quantile(adm, 0.5),
            "ionm_abs_diff_Q3": np.quantile(adm, 0.75),
            "ionm_mean_abs_diff": np.mean(adm),
            "ionm_std_abs_diff": np.std(adm),
            "ionm_dotprod": np.dot(targ_m, pred_m),
            "ionm_cos": _cos_numba(targ_m, pred_m),
        }

        return {k: (0.0 if np.isnan(v) else float(v)) for k, v in features.items()}


def process_spec_batch(spec_batch, sid2idx, h5_path, predictor_name="Intpredict"):
    results = {}
    with h5py.File(h5_path, "r") as h5f:
        intpred_ds = h5f[predictor_name]
        train_ds = h5f["train_data"]
        lengths_ds = h5f["Length"]
        for sid in spec_batch:
            if sid in sid2idx:
                idx = sid2idx[sid]
                intpred = intpred_ds[idx]
                train = train_ds[idx]
                length = int(lengths_ds[idx])
                feats = calculate_ms2pip_features(intpred, train, length)
                results[sid] = feats
            else:
                results[sid] = {}
    return results


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def process_spectra_multithreaded(
    spec_ids_list,
    sid2idx,
    h5_path,
    max_workers=8,
    batch_size=None,
    predictor_name="Intpredict",
):
    total_specs = len(spec_ids_list)
    if batch_size is None:
        batch_size = max(1, total_specs // (max_workers * 4))
        batch_size = min(batch_size, 2000)

    spec_batches = list(chunk_list(spec_ids_list, batch_size))

    features_all = {}
    pbar = tqdm(total=total_specs, desc="Processing spectra", unit="spec")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_spec_batch, batch, sid2idx, h5_path, predictor_name
            ): batch
            for batch in spec_batches
        }

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            batch_results = future.result()
            features_all.update(batch_results)
            pbar.update(len(batch))

    pbar.close()
    return features_all


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", "-i", type=str, required=True)
    parser.add_argument("--tsv_path", "-t", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--predictor_name", type=str, default="Intpredict")
    parser.add_argument("--model_name", type=str, default="mamba")
    parser.add_argument("--workers", "-w", type=int, default=16)
    parser.add_argument("--batch_size", "-b", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_specid_dataset(h5_path: str) -> bool:
    with h5py.File(h5_path, "r") as h5f:
        if "SpecId" in h5f:
            return True

    if _add_specid_to_h5 is None:
        print("H5 missing SpecId and add_specid_to_h5 module not available")
        return False

    _add_specid_to_h5(h5_path=h5_path, overwrite=True, dry_run=False)

    with h5py.File(h5_path, "r") as h5f:
        return "SpecId" in h5f


def validate_files(h5_path, tsv_path, predictor_name: str):
    if not os.path.isfile(h5_path):
        print(f"H5 file not found: {h5_path}")
        return False

    if not os.path.isfile(tsv_path):
        print(f"TSV file not found: {tsv_path}")
        return False

    with h5py.File(h5_path, "r") as h5f:
        required_datasets = ["SpecId", predictor_name, "train_data", "Length"]
        missing_datasets = [ds for ds in required_datasets if ds not in h5f]
        if missing_datasets:
            print(f"H5 missing datasets: {missing_datasets}")
            return False

    df = pd.read_csv(tsv_path, sep="\t", nrows=5)
    if "SpecId" not in df.columns:
        print("TSV missing 'SpecId' column")
        return False

    return True


def main():
    args = parse_arguments()

    if not ensure_specid_dataset(args.h5_path):
        sys.exit(1)

    if not validate_files(args.h5_path, args.tsv_path, predictor_name=args.predictor_name):
        sys.exit(1)

    if args.dry_run:
        sys.exit(0)

    if args.output:
        output_path = args.output
    else:
        workdir = os.path.dirname(os.path.abspath(args.tsv_path))
        if args.model_name == "mamba" or args.predictor_name == "Intpredict":
            output_path = os.path.join(workdir, "msms_specid_with_MS2PIP_m.tsv")
        else:
            output_path = os.path.join(
                workdir, f"{args.model_name}_msms_specid_with_MS2PIP_m.tsv"
            )

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(args.tsv_path, sep="\t")

    with h5py.File(args.h5_path, "r") as h5f:
        spec_ids = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f["SpecId"][:]
        ]

    sid2idx = {sid: i for i, sid in enumerate(spec_ids)}

    spec_ids_list = df["SpecId"].astype(str).tolist()
    matched_count = sum(1 for sid in spec_ids_list if sid in sid2idx)

    if matched_count == 0:
        print("Warning: no matching SpecIds found")

    features_all = process_spectra_multithreaded(
        spec_ids_list,
        sid2idx,
        args.h5_path,
        args.workers,
        args.batch_size,
        args.predictor_name,
    )

    if features_all:
        feat_names = []
        for feats in features_all.values():
            if feats:
                feat_names = list(feats.keys())
                break

        for feat in feat_names:
            df[feat] = df["SpecId"].map(
                lambda sid: features_all.get(sid, {}).get(feat, 0.0)
            )

    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
