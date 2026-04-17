from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

try:
    from .metadata_vocab import (
        METADATA_CONTRACT_VERSION,
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_FRAGMENTATIONS,
        SUPPORTED_MAX_LENGTH,
        TARGET_OUTPUT_SHAPE,
        normalize_collision_energy,
        normalize_fragmentation,
    )
except ImportError:  # pragma: no cover
    from metadata_vocab import (
        METADATA_CONTRACT_VERSION,
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_FRAGMENTATIONS,
        SUPPORTED_MAX_LENGTH,
        TARGET_OUTPUT_SHAPE,
        normalize_collision_energy,
        normalize_fragmentation,
    )


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def inspect_h5_balance(path: str | Path, *, chunk_size: int = 100_000) -> dict:
    charge_counter: Counter = Counter()
    ce_counter: Counter = Counter()
    frag_counter: Counter = Counter()
    charge_frag_counter: Counter = Counter()
    frag_ce_counter: Counter = Counter()
    violations = Counter()
    length_buckets = Counter({"<=30": 0, "31_40": 0, ">40": 0})

    with h5py.File(path, "r") as handle:
        total_rows = int(handle["Charge"].shape[0])
        has_train_data = "train_data" in handle
        train_data_shape = list(handle["train_data"].shape) if has_train_data else None

        if not has_train_data:
            violations["missing_train_data"] += 1
        elif tuple(train_data_shape[1:]) != TARGET_OUTPUT_SHAPE:
            violations["train_data_shape_mismatch"] += 1

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            charges = np.asarray(handle["Charge"][start:end]).astype(np.int64)
            ces_raw = np.asarray(handle["collision_energy"][start:end])
            frags_raw = np.asarray(handle["Fragmentation"][start:end])
            lengths = np.asarray(handle["Length"][start:end]).astype(np.int64)

            frags = [normalize_fragmentation(v) for v in frags_raw]
            normalized_ces = []
            malformed_ce_count = 0
            for value in ces_raw:
                try:
                    normalized_ces.append(normalize_collision_energy(value))
                except (TypeError, ValueError):
                    malformed_ce_count += 1
                    normalized_ces.append(f"MALFORMED:{value}")

            charge_counter.update(charges.tolist())
            ce_counter.update(normalized_ces)
            frag_counter.update(frags)
            charge_frag_counter.update(zip(charges.tolist(), frags))
            frag_ce_counter.update(zip(frags, normalized_ces))

            violations["charge_not_supported"] += int(
                (~np.isin(charges, SUPPORTED_CHARGES)).sum()
            )
            violations["ce_not_supported"] += sum(
                ce not in SUPPORTED_COLLISION_ENERGIES for ce in normalized_ces
            )
            violations["length_gt_40"] += int((lengths > SUPPORTED_MAX_LENGTH).sum())

            length_buckets["<=30"] += int((lengths <= 30).sum())
            length_buckets["31_40"] += int(
                ((lengths >= 31) & (lengths <= SUPPORTED_MAX_LENGTH)).sum()
            )
            length_buckets[">40"] += int((lengths > SUPPORTED_MAX_LENGTH).sum())

    return {
        "path": str(path),
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "rows": total_rows,
        "has_train_data": has_train_data,
        "train_data_shape": train_data_shape,
        "charge_distribution": _counter_to_dict(charge_counter),
        "ce_distribution": _counter_to_dict(ce_counter),
        "fragmentation_distribution": _counter_to_dict(frag_counter),
        "charge_fragmentation": _counter_to_dict(charge_frag_counter),
        "fragmentation_ce": _counter_to_dict(frag_ce_counter),
        "contract_violations": {
            key: int(value) for key, value in sorted(violations.items())
        },
        "length_buckets": {key: int(value) for key, value in length_buckets.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计单个训练 H5 的分布和契约问题")
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--output_json")
    args = parser.parse_args()

    report = inspect_h5_balance(args.input_h5)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
