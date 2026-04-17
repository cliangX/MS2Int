from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

try:
    from .metadata_vocab import (
        METADATA_CONTRACT_VERSION,
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_MAX_LENGTH,
        normalize_collision_energy,
        normalize_fragmentation,
    )
except ImportError:  # pragma: no cover
    from metadata_vocab import (
        METADATA_CONTRACT_VERSION,
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_MAX_LENGTH,
        normalize_collision_energy,
        normalize_fragmentation,
    )


def compute_fragment_targets(
    fragment_counts: dict[str, int],
    *,
    target_cid_ratio: float,
    max_upsample_factor: int,
) -> dict[str, int]:
    hcd_count = int(fragment_counts.get("HCD", 0))
    cid_count = int(fragment_counts.get("CID", 0))

    if cid_count <= 0 or target_cid_ratio <= 0:
        return {"HCD": hcd_count, "CID": 0}
    if hcd_count <= 0:
        return {"HCD": 0, "CID": cid_count}

    desired_cid = int(np.ceil(hcd_count * target_cid_ratio / (1.0 - target_cid_ratio)))
    cid_capacity = cid_count * max_upsample_factor

    if desired_cid <= cid_capacity:
        return {"HCD": hcd_count, "CID": desired_cid}

    target_cid = cid_capacity
    target_hcd = int(np.floor(target_cid * (1.0 - target_cid_ratio) / target_cid_ratio))
    return {"HCD": min(hcd_count, target_hcd), "CID": target_cid}


def allocate_targets(
    counts: dict,
    *,
    total_target: int,
    weight_mode: str,
    max_upsample_factor: int,
) -> dict:
    if not counts:
        return {}

    keys = list(counts.keys())
    count_values = np.array([int(counts[key]) for key in keys], dtype=np.int64)
    capacities = count_values * max_upsample_factor
    total_target = int(min(total_target, int(capacities.sum())))

    if weight_mode == "sqrt_count":
        weights = np.sqrt(count_values.astype(np.float64))
    elif weight_mode == "count":
        weights = count_values.astype(np.float64)
    else:
        raise ValueError(f"不支持的 weight_mode: {weight_mode}")

    targets = np.zeros(len(keys), dtype=np.int64)
    remaining_target = total_target
    active = np.ones(len(keys), dtype=bool)

    while remaining_target > 0 and active.any():
        active_weights = weights[active]
        raw = remaining_target * active_weights / active_weights.sum()
        active_indices = np.flatnonzero(active)
        capped_any = False

        for active_pos, index in enumerate(active_indices):
            cap_left = int(capacities[index] - targets[index])
            if raw[active_pos] >= cap_left:
                targets[index] += cap_left
                remaining_target -= cap_left
                active[index] = False
                capped_any = True

        if capped_any:
            continue

        floors = np.floor(raw).astype(np.int64)
        for active_pos, index in enumerate(active_indices):
            targets[index] += floors[active_pos]
        remaining_target -= int(floors.sum())

        if remaining_target <= 0:
            break

        fractions = raw - floors
        order = np.argsort(-fractions)
        for active_pos in order[:remaining_target]:
            targets[active_indices[active_pos]] += 1
        remaining_target = 0

    return {key: int(targets[idx]) for idx, key in enumerate(keys)}


def _collect_groups(
    input_h5: str | Path,
    *,
    allowed_ce: set[int],
    max_length: int,
    drop_charge: set[int],
    chunk_size: int = 100_000,
) -> dict[tuple[str, int, int], np.ndarray]:
    groups: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    with h5py.File(input_h5, "r") as handle:
        total_rows = int(handle["Charge"].shape[0])
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            charges = np.asarray(handle["Charge"][start:end]).astype(np.int64)
            ces_raw = np.asarray(handle["collision_energy"][start:end])
            lengths = np.asarray(handle["Length"][start:end]).astype(np.int64)
            frags_raw = np.asarray(handle["Fragmentation"][start:end])

            frags = np.array(
                [normalize_fragmentation(v) for v in frags_raw], dtype=object
            )
            ces = np.array(
                [normalize_collision_energy(v) for v in ces_raw], dtype=np.int64
            )
            valid_mask = (
                (lengths <= max_length)
                & (~np.isin(charges, list(drop_charge)))
                & np.isin(charges, SUPPORTED_CHARGES)
                & np.isin(ces, list(allowed_ce))
            )

            if not np.any(valid_mask):
                continue

            rows = np.nonzero(valid_mask)[0] + start
            charges_v = charges[valid_mask]
            ces_v = ces[valid_mask]
            frags_v = frags[valid_mask]

            combo_keys = np.empty(
                len(rows),
                dtype=[("frag", "U8"), ("charge", np.int64), ("ce", np.int64)],
            )
            combo_keys["frag"] = np.asarray(frags_v, dtype="U8")
            combo_keys["charge"] = charges_v
            combo_keys["ce"] = ces_v
            unique_keys, inverse = np.unique(combo_keys, return_inverse=True)
            for idx, key_values in enumerate(unique_keys):
                group_key = (
                    str(key_values["frag"]),
                    int(key_values["charge"]),
                    int(key_values["ce"]),
                )
                groups[group_key].append(rows[inverse == idx])
    return {key: np.concatenate(chunks) for key, chunks in groups.items()}


def _compute_group_targets(
    groups: dict[tuple[str, int, int], np.ndarray],
    *,
    target_cid_ratio: float,
    charge_balance_mode: str,
    max_upsample_factor: int,
) -> dict[tuple[str, int, int], int]:
    group_counts = {key: len(rows) for key, rows in groups.items()}
    fragment_counts = Counter()
    fragment_charge_counts = Counter()
    ce_counts_by_fragment_charge: dict[tuple[str, int], dict[int, int]] = defaultdict(
        dict
    )

    for (frag, charge, ce), count in group_counts.items():
        fragment_counts[frag] += count
        fragment_charge_counts[(frag, charge)] += count
        ce_counts_by_fragment_charge[(frag, charge)][ce] = count

    fragment_targets = compute_fragment_targets(
        dict(fragment_counts),
        target_cid_ratio=target_cid_ratio,
        max_upsample_factor=max_upsample_factor,
    )

    final_targets: dict[tuple[str, int, int], int] = {}
    for frag, frag_total_target in fragment_targets.items():
        charge_counts = {
            charge: count
            for (frag_name, charge), count in fragment_charge_counts.items()
            if frag_name == frag
        }
        charge_targets = allocate_targets(
            charge_counts,
            total_target=frag_total_target,
            weight_mode=charge_balance_mode,
            max_upsample_factor=max_upsample_factor,
        )
        for charge, charge_target in charge_targets.items():
            ce_targets = allocate_targets(
                ce_counts_by_fragment_charge[(frag, charge)],
                total_target=charge_target,
                weight_mode="count",
                max_upsample_factor=max_upsample_factor,
            )
            for ce, ce_target in ce_targets.items():
                final_targets[(frag, charge, ce)] = ce_target

    return final_targets


def _sample_rows(
    groups: dict[tuple[str, int, int], np.ndarray],
    group_targets: dict[tuple[str, int, int], int],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for key, rows in groups.items():
        target = int(group_targets.get(key, 0))
        if target <= 0:
            continue
        sampled = rng.choice(rows, size=target, replace=(target > len(rows)))
        selected.append(np.asarray(sampled, dtype=np.int64))
    if not selected:
        return np.array([], dtype=np.int64)
    return np.concatenate(selected)


def _write_selected_rows(
    input_h5: str | Path, output_h5: str | Path, selected_rows: np.ndarray
) -> None:
    counts = Counter(selected_rows.tolist())
    unique_rows = np.array(sorted(counts), dtype=np.int64)
    repeats = np.array([counts[row] for row in unique_rows], dtype=np.int64)

    with h5py.File(input_h5, "r") as source, h5py.File(output_h5, "w") as target:
        total_rows = int(repeats.sum())
        for key in source.keys():
            source_ds = source[key]
            target_shape = (total_rows,) + tuple(source_ds.shape[1:])
            target_ds = target.create_dataset(
                key, shape=target_shape, dtype=source_ds.dtype
            )

            write_offset = 0
            chunk_size = 10_000
            for start in range(0, len(unique_rows), chunk_size):
                end = min(start + chunk_size, len(unique_rows))
                rows_chunk = unique_rows[start:end]
                counts_chunk = repeats[start:end]
                data = source_ds[rows_chunk]
                repeated = np.repeat(data, counts_chunk, axis=0)
                target_ds[write_offset : write_offset + len(repeated)] = repeated
                write_offset += len(repeated)


def rebalance_single_h5(
    *,
    input_h5: str | Path,
    output_path: str | Path,
    target_cid_ratio: float = 0.15,
    charge_balance_mode: str = "sqrt_count",
    allowed_ce: tuple[int, ...] = SUPPORTED_COLLISION_ENERGIES,
    max_length: int = SUPPORTED_MAX_LENGTH,
    drop_charge: tuple[int, ...] = (7,),
    max_upsample_factor: int = 4,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    groups = _collect_groups(
        input_h5,
        allowed_ce=set(allowed_ce),
        max_length=max_length,
        drop_charge=set(drop_charge),
    )
    group_targets = _compute_group_targets(
        groups,
        target_cid_ratio=target_cid_ratio,
        charge_balance_mode=charge_balance_mode,
        max_upsample_factor=max_upsample_factor,
    )
    selected_rows = np.sort(
        _sample_rows(groups, group_targets, seed=seed).astype(np.int64)
    )

    result = {
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "selected_rows": selected_rows,
        "group_targets": {
            str(key): int(value) for key, value in sorted(group_targets.items())
        },
    }

    if not dry_run:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_selected_rows(input_h5, output_path, selected_rows)
        manifest = {
            "input_h5": str(input_h5),
            "output_path": str(output_path),
            "target_fragmentation_ratio": {
                "HCD": 1.0 - target_cid_ratio,
                "CID": target_cid_ratio,
            },
            "charge_balance_mode": charge_balance_mode,
            "allowed_ce": list(allowed_ce),
            "max_length": max_length,
            "max_upsample_factor": max_upsample_factor,
            "selected_rows": int(len(selected_rows)),
            "metadata_contract_version": METADATA_CONTRACT_VERSION,
        }
        output_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return result


def _parse_allowed_ce(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对单个训练 H5 进行重采样并生成 balanced_train.h5"
    )
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_cid_ratio", type=float, default=0.15)
    parser.add_argument("--charge_balance_mode", default="sqrt_count")
    parser.add_argument("--drop_charge", type=int, action="append", default=[])
    parser.add_argument(
        "--allowed_ce", default=",".join(str(v) for v in SUPPORTED_COLLISION_ENERGIES)
    )
    parser.add_argument("--max_length", type=int, default=SUPPORTED_MAX_LENGTH)
    parser.add_argument("--max_upsample_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    result = rebalance_single_h5(
        input_h5=args.input_h5,
        output_path=args.output,
        target_cid_ratio=args.target_cid_ratio,
        charge_balance_mode=args.charge_balance_mode,
        allowed_ce=_parse_allowed_ce(args.allowed_ce),
        max_length=args.max_length,
        drop_charge=tuple(args.drop_charge),
        max_upsample_factor=args.max_upsample_factor,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    printable = {
        "metadata_contract_version": result["metadata_contract_version"],
        "selected_rows": int(len(result["selected_rows"])),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
