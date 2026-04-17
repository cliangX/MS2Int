from __future__ import annotations

from typing import Any


SUPPORTED_CHARGES = (1, 2, 3, 4, 5, 6)
SUPPORTED_COLLISION_ENERGIES = (25, 26, 27, 28, 30, 32, 35)
SUPPORTED_FRAGMENTATIONS = ("HCD", "CID")
SUPPORTED_MAX_LENGTH = 40
TARGET_OUTPUT_SHAPE = (39, 41)
METADATA_CONTRACT_VERSION = "v3-train40-output39x41"
SOURCE_ROW_ID_DATASET = "source_row_id"

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
