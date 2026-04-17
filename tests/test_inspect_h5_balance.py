from __future__ import annotations

from MS2Int.inspect_h5_balance import inspect_h5_balance

from .conftest import write_h5_rows


def test_inspect_reports_shape_and_contract_violations(tmp_path):
    train_path = tmp_path / "train.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "HCD",
            "annotate": "PEPTIDE",
            "Length": 40,
        },
        {
            "Raw_file": "raw2",
            "MS2_Scan_Number": 2,
            "Charge": 7,
            "collision_energy": 42.0,
            "Fragmentation": "CID",
            "annotate": "PEPTIDER",
            "Length": 41,
        },
    ]

    write_h5_rows(train_path, rows, include_train_data=True)

    report = inspect_h5_balance(train_path)

    assert report["rows"] == 2
    assert report["train_data_shape"] == [2, 39, 41]
    assert report["contract_violations"]["charge_not_supported"] == 1
    assert report["contract_violations"]["ce_not_supported"] == 1
    assert report["contract_violations"]["length_gt_40"] == 1
    assert report["length_buckets"]["31_40"] == 1


def test_inspect_does_not_crash_on_malformed_collision_energy(tmp_path):
    train_path = tmp_path / "train_bad_ce.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": "bad_ce",
            "Fragmentation": "HCD",
            "annotate": "PEPTIDE",
            "Length": 12,
        }
    ]

    write_h5_rows(train_path, rows, include_train_data=True)

    report = inspect_h5_balance(train_path)

    assert report["contract_violations"]["ce_not_supported"] == 1


def test_inspect_flags_wrong_train_data_shape(tmp_path):
    train_path = tmp_path / "train_shape.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "HCD",
            "annotate": "PEPTIDE",
            "Length": 12,
        }
    ]

    write_h5_rows(train_path, rows, include_train_data=True, train_shape=(29, 31))

    report = inspect_h5_balance(train_path)

    assert report["contract_violations"]["train_data_shape_mismatch"] == 1


def test_inspect_flags_missing_train_data(tmp_path):
    train_path = tmp_path / "train_missing_train.h5"

    rows = [
        {
            "Raw_file": "raw1",
            "MS2_Scan_Number": 1,
            "Charge": 2,
            "collision_energy": 25.0,
            "Fragmentation": "HCD",
            "annotate": "PEPTIDE",
            "Length": 12,
        }
    ]

    write_h5_rows(train_path, rows, include_train_data=False)

    report = inspect_h5_balance(train_path)

    assert report["contract_violations"]["missing_train_data"] == 1
