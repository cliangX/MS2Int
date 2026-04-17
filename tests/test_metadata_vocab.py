from __future__ import annotations

import pytest

from MS2Int.metadata_vocab import (
    TARGET_OUTPUT_SHAPE,
    encode_charge,
    encode_collision_energy,
    validate_length,
)


def test_charge_7_must_be_rejected():
    with pytest.raises(ValueError, match="Charge"):
        encode_charge(7)


def test_ce_32_must_be_supported():
    assert encode_collision_energy(32) == 5


def test_ce_29_must_be_rejected_after_real_audit():
    with pytest.raises(ValueError, match="collision energy"):
        encode_collision_energy(29)


def test_ce_42_must_be_rejected():
    with pytest.raises(ValueError, match="collision energy"):
        encode_collision_energy(42)


def test_length_40_must_be_supported_end_to_end():
    assert validate_length(40) == 40
    assert TARGET_OUTPUT_SHAPE == (39, 41)
