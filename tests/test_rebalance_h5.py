from __future__ import annotations

import numpy as np

from MS2Int.rebalance_h5 import (
    allocate_targets,
    compute_fragment_targets,
    rebalance_single_h5,
)

from .conftest import write_h5_rows


def test_rebalance_caps_majority_and_minority_sampling():
    targets = compute_fragment_targets(
        {"HCD": 100, "CID": 10}, target_cid_ratio=0.15, max_upsample_factor=4
    )

    assert targets == {"HCD": 100, "CID": 18}

    charge_targets = allocate_targets(
        {2: 100, 3: 25, 4: 4},
        total_target=60,
        weight_mode="sqrt_count",
        max_upsample_factor=4,
    )

    assert sum(charge_targets.values()) == 60
    assert charge_targets[2] > charge_targets[3] > charge_targets[4]
    assert charge_targets[4] <= 16


def test_rebalance_preserves_ce_distribution_within_stratum():
    ce_targets = allocate_targets(
        {25: 8, 30: 2},
        total_target=5,
        weight_mode="count",
        max_upsample_factor=4,
    )

    assert sum(ce_targets.values()) == 5
    assert ce_targets[25] == 4
    assert ce_targets[30] == 1


def test_rebalance_single_h5_preserves_source_row_id_when_present(tmp_path):
    train_path = tmp_path / "train.h5"
    output_path = tmp_path / "balanced_train.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "HCD",
            "annotate": "PEPA",
            "Length": 10,
            "source_row_id": 100,
        },
        {
            "Raw_file": "raw2",
            "MS2_Scan_Number": 2,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "CID",
            "annotate": "PEPB",
            "Length": 10,
            "source_row_id": 101,
        },
    ]

    write_h5_rows(train_path, rows, include_train_data=True)

    result = rebalance_single_h5(
        input_h5=train_path,
        output_path=output_path,
        target_cid_ratio=0.15,
        charge_balance_mode="sqrt_count",
        dry_run=False,
        seed=7,
    )

    assert result["selected_rows"].tolist() == [0, 1]
    assert output_path.exists()

    import h5py

    with h5py.File(output_path, "r") as f:
        assert "source_row_id" in f
        assert f["source_row_id"][:].tolist() == [100, 101]


def test_rebalance_single_h5_works_without_source_row_id(tmp_path):
    train_path = tmp_path / "train_no_source.h5"
    output_path = tmp_path / "balanced_no_source.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "HCD",
            "annotate": "PEPA",
            "Length": 10,
        },
        {
            "Raw_file": "raw2",
            "MS2_Scan_Number": 2,
            "Charge": 3,
            "collision_energy": 30.0,
            "Fragmentation": "CID",
            "annotate": "PEPB",
            "Length": 10,
        },
    ]

    write_h5_rows(train_path, rows, include_train_data=True)

    result = rebalance_single_h5(
        input_h5=train_path,
        output_path=output_path,
        target_cid_ratio=0.15,
        charge_balance_mode="sqrt_count",
        dry_run=False,
        seed=7,
    )

    assert result["selected_rows"].tolist() == [0, 1]
    assert output_path.exists()

    import h5py

    with h5py.File(output_path, "r") as f:
        assert "source_row_id" not in f
