from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _to_h5_array(values):
    if not values:
        return np.array([], dtype=np.int64)

    first = values[0]
    if isinstance(first, str):
        width = max(len(v.encode("utf-8")) for v in values)
        return np.array(values, dtype=f"S{max(width, 1)}")
    if isinstance(first, bytes):
        width = max(len(v) for v in values)
        return np.array(values, dtype=f"S{max(width, 1)}")
    if isinstance(first, float):
        return np.array(values, dtype=np.float64)
    return np.array(values, dtype=np.int64)


def write_h5_rows(
    path: Path,
    rows: list[dict],
    *,
    include_train_data: bool = False,
    train_shape=(39, 41),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = rows[0].keys() if rows else []
    with h5py.File(path, "w") as handle:
        for column in columns:
            handle.create_dataset(
                column, data=_to_h5_array([row[column] for row in rows])
            )

        if include_train_data:
            train_data = np.zeros((len(rows),) + tuple(train_shape), dtype=np.float32)
            handle.create_dataset("train_data", data=train_data)
