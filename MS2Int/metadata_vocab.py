"""Train40 数据准备顺序：0/2 — 训练元数据契约与编码；被 preprocess、step1_h5_audit、step2_h5_rebalance 等导入，通常不单独作为 CLI 运行。"""

from __future__ import annotations

from typing import Any


SUPPORTED_CHARGES = (1, 2, 3, 4, 5, 6)
SUPPORTED_COLLISION_ENERGIES = (25, 26, 27, 28, 30, 32, 35)
SUPPORTED_FRAGMENTATIONS = ("HCD", "CID")
SUPPORTED_MAX_LENGTH = 40
TARGET_OUTPUT_SHAPE = (39, 41)
METADATA_CONTRACT_VERSION = "v3-train40-output39x41"
SOURCE_ROW_ID_DATASET = "source_row_id"

# AA 词表：0 保留给 padding，1..54 为有效 token
# 顺序必须与历史 enumerate(AA) 一致，否则会破坏已有 checkpoint
SUPPORTED_AA_VOCAB = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
    "[]-": 21, "-[]": 22,
    "[Acetyl]-": 23,
    "M[Oxidation]": 24, "S[Phospho]": 25, "T[Phospho]": 26,
    "Y[Phospho]": 27, "K[Dimethyl]": 28, "K[Trimethyl]": 29,
    "K[Formyl]": 30, "K[Propionyl]": 31, "K[Succinyl]": 32,
    "K[Biotin]": 33, "K[UNIMOD:737]": 34, "R[Dimethyl]": 35,
    "R[UNIMOD:36a]": 36, "P[Oxidation]": 37, "Y[Nitro]": 38,
    "K[Methyl]": 39, "T[HexNAc]": 40, "S[HexNAc]": 41,
    "C[Carbamidomethyl]": 42, "E[Glu->pyro-Glu]": 43,
    "R[Phospho]": 44, "K[Acetyl]": 45, "K[GG]": 46,
    "Q[Gln->pyro-Glu]": 47, "R[Methyl]": 48,
    "[UNIMOD:737]-": 49, "K[UNIMOD:1848]": 50,
    "K[UNIMOD:1363]": 51, "K[UNIMOD:1849]": 52,
    "K[UNIMOD:1289]": 53, "K[UNIMOD:747]": 54,
}
NUM_AA_TOKENS = len(SUPPORTED_AA_VOCAB)  # 54；Embedding 大小应为 NUM_AA_TOKENS + 1

_CHARGE_TO_INDEX = {charge: index for index, charge in enumerate(SUPPORTED_CHARGES)}
_CE_TO_INDEX = {
    energy: index for index, energy in enumerate(SUPPORTED_COLLISION_ENERGIES)
}
_FRAG_TO_INDEX = {frag: index for index, frag in enumerate(SUPPORTED_FRAGMENTATIONS)}


def _coerce_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _normalize_collision_energy(value: Any) -> int:
    value = _coerce_scalar(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"collision energy 必须是整数，收到 {value!r}")
    return int(value)


def encode_aa(token: str, strict: bool = True) -> int:
    """将 AA token 编码为索引。未知 token 在 strict 模式下抛错，否则返回 -1。"""
    idx = SUPPORTED_AA_VOCAB.get(token)
    if idx is not None:
        return idx
    if strict:
        raise ValueError(
            f"AA token 不受支持: {token!r}。"
            f"若为新修饰，请先将其添加到 metadata_vocab.SUPPORTED_AA_VOCAB"
        )
    return -1


def encode_charge(charge: Any, strict: bool = True) -> int:
    charge = int(_coerce_scalar(charge))
    if charge not in _CHARGE_TO_INDEX:
        if strict:
            raise ValueError(f"Charge 不受支持: {charge}")
        return -1
    return _CHARGE_TO_INDEX[charge]


def encode_collision_energy(ce: Any, strict: bool = True) -> int:
    ce = _normalize_collision_energy(ce)
    if ce not in _CE_TO_INDEX:
        if strict:
            raise ValueError(f"collision energy 不受支持: {ce}")
        return -1
    return _CE_TO_INDEX[ce]


def encode_fragmentation(frag: Any, strict: bool = True) -> int:
    frag = str(_coerce_scalar(frag)).upper()
    if frag not in _FRAG_TO_INDEX:
        if strict:
            raise ValueError(f"Fragmentation 不受支持: {frag}")
        return -1
    return _FRAG_TO_INDEX[frag]


def validate_length(
    length: Any, max_length: int = SUPPORTED_MAX_LENGTH, strict: bool = True
) -> int:
    length = int(_coerce_scalar(length))
    if length < 0:
        raise ValueError(f"Length 不能为负数: {length}")
    if length > max_length:
        if strict:
            raise ValueError(f"Length 超出支持范围: {length} > {max_length}")
        return max_length
    return length


def normalize_fragmentation(value: Any) -> str:
    return str(_coerce_scalar(value)).upper()


def normalize_collision_energy(value: Any) -> int:
    return _normalize_collision_energy(value)


def normalize_key_value(value: Any) -> Any:
    value = _coerce_scalar(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
