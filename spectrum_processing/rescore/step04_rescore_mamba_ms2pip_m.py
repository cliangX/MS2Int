#!/usr/bin/env python3
"""Two-stage mokapot rescoring with external TSV features and optional Basic/MaxQuant features."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

if os.getenv("MS2INT_ENABLE_TF_GPU", "0").lower() not in ("1", "true", "yes"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from mokapot import brew
from mokapot.model import PercolatorModel

from ms2rescore.feature_generators.basic import BasicFeatureGenerator
from ms2rescore.feature_generators.maxquant import MaxQuantFeatureGenerator
from ms2rescore.rescoring_engines.mokapot import convert_psm_list
from psm_utils.io import read_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--msms_path", required=True)
    p.add_argument("-t", "--tsv_path", required=True)
    p.add_argument("--drop_specid", action="store_true", default=True)
    p.add_argument("--rng", type=int, default=42)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--train_fdr", type=float, default=0.01)
    p.add_argument("--test_fdr", type=float, default=0.01)
    p.add_argument("--no-basic", dest="add_basic", action="store_false")
    p.add_argument("--no-maxquant", dest="add_maxquant", action="store_false")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--brew-log-level", default="WARNING",
                   choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return p.parse_args()


def convert_value(colname: str, value: Any) -> Any:
    if colname.startswith("charge_"):
        return np.int8(value)
    if colname == "charge_n":
        return np.int64(value)
    if isinstance(value, (int, np.integer)):
        return np.int64(value)
    return np.float64(value)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(pd, "Timedelta") and isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(x) for x in list(obj)]
    return obj


def accepted_to_records(accepted_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(accepted_obj, dict):
        if all(not isinstance(v, (list, tuple, set, np.ndarray, pd.Series)) for v in accepted_obj.values()):
            return [_to_builtin(accepted_obj)]
    df = pd.DataFrame(accepted_obj)
    if not df.empty:
        return [_to_builtin(r) for r in df.to_dict(orient="records")]
    if isinstance(accepted_obj, list):
        return [_to_builtin(x) for x in accepted_obj]
    return [_to_builtin(accepted_obj)]


def _brew_with_train_fdr_fallback(
    ds: Any,
    *,
    rng: int,
    folds: int,
    max_workers: int,
    train_fdr: float,
    test_fdr: float,
    logger: logging.Logger,
):
    if folds < 2:
        raise ValueError("--folds must be >= 2")

    train_candidates: List[float] = []
    for fdr in (train_fdr, 0.05, 0.1, 0.5, 1.0):
        if fdr not in train_candidates:
            train_candidates.append(float(fdr))

    test_candidates: List[float] = []
    for fdr in (test_fdr, 0.05, 0.1, 0.5, 1.0):
        if fdr not in test_candidates:
            test_candidates.append(float(fdr))

    last_err: Exception | None = None
    for train in train_candidates:
        model = PercolatorModel(train_fdr=float(train), rng=rng)
        for test in test_candidates:
            try:
                return brew(
                    ds,
                    model=model,
                    test_fdr=float(test),
                    folds=int(folds),
                    max_workers=int(max_workers),
                    rng=rng,
                )
            except RuntimeError as e:
                msg = str(e)
                last_err = e

                # Label init failed: train_fdr too strict
                if "No PSMs found below the 'eval_fdr'" in msg:
                    if train != train_candidates[-1]:
                        logger.warning(
                            f"brew failed (train_fdr={train:g}): {msg}; trying looser train_fdr"
                        )
                    break

                # Calibration failed: test_fdr too strict
                if (
                    "Failed to calibrate scores between cross-validation folds" in msg
                    and "Try raising 'test_fdr'" in msg
                ) or ("No target PSMs were below the 'eval_fdr' threshold" in msg):
                    if test != test_candidates[-1]:
                        logger.warning(
                            f"brew failed (test_fdr={test:g}): {msg}; trying looser test_fdr"
                        )
                        continue
                    raise

                raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("mokapot brew failed")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("rescore_mamba_m")

    def set_library_log_level(logger_name: str, level_name: str) -> None:
        import logging as _logging

        level = getattr(_logging, str(level_name).upper(), _logging.WARNING)
        lg = _logging.getLogger(logger_name)
        lg.setLevel(level)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)
        if level >= _logging.ERROR:
            lg.addHandler(_logging.NullHandler())

    level_from_env = os.getenv("MOKAPOT_LOG_LEVEL", args.brew_log_level)
    set_library_log_level("mokapot", level_from_env)

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    logger.info(f"Reading PSMs: {args.msms_path}")
    psm_list = read_file(args.msms_path, filetype="msms")

    psm_list.rename_modifications({"ox": "Oxidation"})
    psm_list.add_fixed_modifications([("U:Carbamidomethyl", ["C"])])
    psm_list.apply_fixed_modifications()

    logger.info(f"Reading features: {args.tsv_path}")
    df_raw = pd.read_csv(args.tsv_path, sep="\t")
    if args.drop_specid and "SpecId" in df_raw.columns:
        df_raw = df_raw.drop(columns=["SpecId"])

    df = df_raw.select_dtypes(include=[np.number])
    dropped_cols = [c for c in df_raw.columns if c not in df.columns]
    if dropped_cols:
        logger.info(f"Kept {len(df.columns)} numeric cols, dropped {len(dropped_cols)} non-numeric")

    if len(df) != len(psm_list):
        raise ValueError(f"TSV rows ({len(df)}) != PSM count ({len(psm_list)})")

    logger.info("Building rescoring_features ...")
    records: List[Dict[str, Any]] = []
    cols = list(df.columns)
    for _, row in df.iterrows():
        row_dict = {col: convert_value(col, row[col]) for col in cols}
        records.append(row_dict)
    psm_list["rescoring_features"] = records

    if not args.verbose:
        logging.getLogger("mokapot").setLevel(logging.WARNING)

    logger.info(
        f"Phase 1: brew(rng={args.rng}, folds={args.folds}, max_workers={args.max_workers})"
    )
    ds1 = convert_psm_list(psm_list)
    conf1, _ = _brew_with_train_fdr_fallback(
        ds1,
        rng=args.rng,
        folds=args.folds,
        max_workers=args.max_workers,
        train_fdr=args.train_fdr,
        test_fdr=args.test_fdr,
        logger=logger,
    )

    did_add = []
    if args.add_basic:
        logger.info("Phase 2: adding Basic features")
        BasicFeatureGenerator().add_features(psm_list)
        did_add.append("basic")
    if args.add_maxquant:
        logger.info("Phase 2: adding MaxQuant features")
        MaxQuantFeatureGenerator().add_features(psm_list)
        did_add.append("maxquant")

    final_conf = conf1
    if did_add:
        logger.info(
            f"Phase 2: brew(rng={args.rng}, folds={args.folds}, max_workers={args.max_workers})"
        )
        ds2 = convert_psm_list(psm_list)
        conf2, _ = _brew_with_train_fdr_fallback(
            ds2,
            rng=args.rng,
            folds=args.folds,
            max_workers=args.max_workers,
            train_fdr=args.train_fdr,
            test_fdr=args.test_fdr,
            logger=logger,
        )
        final_conf = conf2

    export_dir = os.path.join(os.getcwd(), "rescore", "mokapot")
    os.makedirs(export_dir, exist_ok=True)
    logger.info(f"Exporting mokapot results to: {export_dir}")
    final_conf.to_txt(dest_dir=export_dir)

    records_out = accepted_to_records(final_conf.accepted)

    logger.info(f"Done: accepted={len(records_out)}, output: {export_dir}")


if __name__ == "__main__":
    main()
