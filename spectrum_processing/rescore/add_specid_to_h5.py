#!/usr/bin/env python3
"""Generate and write SpecId dataset into an HDF5 file."""

import argparse
import logging
import os
import sys
from typing import List

import h5py
import numpy as np


def _decode_str_array(arr) -> List[str]:
    out: List[str] = []
    for v in arr:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def _to_int_str_array(arr) -> List[str]:
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.integer):
            return [str(int(x)) for x in arr]
        if np.issubdtype(arr.dtype, np.floating):
            return [str(int(round(float(x)))) for x in arr]
    out: List[str] = []
    for v in arr:
        out.append(str(int(round(float(v)))))
    return out


def build_specid(
    raw_files: List[str],
    scans: List[str],
    seqs: List[str],
    charges: List[str],
    replace_carba: bool = True,
) -> List[str]:
    if replace_carba:
        seqs = [s.replace("[Carbamidomethyl]", "[UNIMOD:4]") for s in seqs]
    return [
        f"{rf}-{msn}-{seq}-{ch}"
        for rf, msn, seq, ch in zip(raw_files, scans, seqs, charges)
    ]


def add_specid_to_h5(
    h5_path: str,
    raw_key: str = "Raw_file",
    scan_key: str = "MS2_Scan_Number",
    seq_key: str = "annotate",
    charge_key: str = "Charge",
    replace_carba: bool = True,
    overwrite: bool = True,
    dry_run: bool = False,
) -> None:
    logger = logging.getLogger("add_SpecId2h5")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, "a") as h5f:
        for k in [raw_key, scan_key, seq_key, charge_key]:
            if k not in h5f:
                raise KeyError(f"H5 missing dataset: {k}")

        raw_arr = h5f[raw_key][:]
        seq_arr = h5f[seq_key][:]
        scan_arr = h5f[scan_key][:]
        charge_arr = h5f[charge_key][:]

        raw_list = _decode_str_array(raw_arr)
        seq_list = _decode_str_array(seq_arr)
        scan_list = _to_int_str_array(scan_arr)
        charge_list = _to_int_str_array(charge_arr)

        n = len(raw_list)
        if not (len(seq_list) == len(scan_list) == len(charge_list) == n):
            raise ValueError(
                f"Length mismatch: Raw={len(raw_list)}, Scan={len(scan_list)}, Seq={len(seq_list)}, Charge={len(charge_list)}"
            )

        specids = build_specid(
            raw_list, scan_list, seq_list, charge_list, replace_carba=replace_carba
        )

        if dry_run:
            return

        if "SpecId" in h5f:
            if overwrite:
                del h5f["SpecId"]
            else:
                return

        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset(
            "SpecId", data=np.array(specids, dtype=object), dtype=str_dtype
        )
        logger.info(f"Wrote {n} SpecIds -> {h5_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--h5_path", required=True)
    p.add_argument("--raw_key", default="Raw_file")
    p.add_argument("--scan_key", default="MS2_Scan_Number")
    p.add_argument("--seq_key", default="annotate")
    p.add_argument("--charge_key", default="Charge")
    p.add_argument("--no-replace-carba", dest="replace_carba", action="store_false")
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    add_specid_to_h5(
        h5_path=args.h5_path,
        raw_key=args.raw_key,
        scan_key=args.scan_key,
        seq_key=args.seq_key,
        charge_key=args.charge_key,
        replace_carba=args.replace_carba,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
