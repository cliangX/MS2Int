"""Train40 离子注释矩阵与强度向量化（FLR 磷酸化参考谱 / cosine 对齐用）。"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

MAX_PEPTIDE_LENGTH = 40
ION_ROWS = MAX_PEPTIDE_LENGTH + 1
ION_COLS = MAX_PEPTIDE_LENGTH - 1
# H5 中 train_data / Intpredict 的存储形状（step3 swapaxes 之后）
H5_SPECTRUM_SHAPE = (ION_COLS, ION_ROWS)


def _build_annotation():
    mat = np.full((ION_ROWS, ION_COLS), "", dtype=object)
    for i in range(ION_COLS):
        mat[0, i] = f"b{i + 1}"
    for i in range(ION_COLS):
        mat[1, i] = f"b{i + 1}^2"
    for i in range(ION_COLS):
        mat[2, i] = f"y{i + 1}"
    for i in range(ION_COLS):
        mat[3, i] = f"y{i + 1}^2"
    for row in range(4, ION_ROWS):
        for col in range(ION_COLS):
            m_start = col + 2
            m_end = m_start + (row - 2)
            if m_end <= MAX_PEPTIDE_LENGTH:
                mat[row, col] = f"m{m_start}:{m_end}"

    ion_order = mat.ravel().tolist()
    index = {name: i for i, name in enumerate(ion_order) if name}
    return mat, ion_order, index


ANNOTATION_MATRIX, ION_ORDER, ION_TO_IDX = _build_annotation()


def _normalize_ion_name(name: str):
    if not name:
        return None

    base = str(name)
    charge_suffix = ""
    if "^" in base:
        parts = base.split("^", 1)
        base, charge_suffix = parts[0], "^" + parts[1]

    if base.endswith("-H3PO4"):
        base = base[: -len("-H3PO4")]

    if not base:
        return None

    if base.startswith("m"):
        return base

    return base + charge_suffix
