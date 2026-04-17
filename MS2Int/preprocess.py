import h5py
import torch
import numpy as np

try:
    from .metadata_vocab import (
        encode_charge,
        encode_collision_energy,
        encode_fragmentation,
        validate_length,
        SUPPORTED_MAX_LENGTH,
    )
except ImportError:  # pragma: no cover
    from metadata_vocab import (
        encode_charge,
        encode_collision_energy,
        encode_fragmentation,
        validate_length,
        SUPPORTED_MAX_LENGTH,
    )


def tokenize_peptide(seq: str):
    tokens = []
    i, n = 0, len(seq)
    if seq.startswith("["):
        j = seq.find("]-", 0)
        if j == -1:
            raise ValueError("N-terminus modification missing ']-'")
        tokens.append(seq[: j + 2])
        i = j + 2
    while i < n:
        if seq.startswith("-[]", i):
            tokens.append("-[]")
            i += 3
            continue
        ch = seq[i]
        if "A" <= ch <= "Z":
            i += 1
            if i < n and seq[i] == "[":
                j = seq.find("]", i)
                if j == -1:
                    raise ValueError("Residue modification missing ']' ")
                tokens.append(ch + seq[i : j + 1])
                i = j + 1
            else:
                tokens.append(ch)
        else:
            raise ValueError(f"Invalid character '{ch}' at pos {i}")
    return tokens


def data_read(h5_or_path, idx, include_train: bool = True):
    need_close = False
    if isinstance(h5_or_path, str):
        f = h5py.File(h5_or_path, "r")
        need_close = True
    else:
        f = h5_or_path

    try:
        if "annotate" in f:
            Sequence = f["annotate"][idx : idx + 1]
        else:
            Sequence = f["Sequence"][idx : idx + 1]
        Sequence = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else s for s in Sequence]
        )
        Length = f["Length"][idx : idx + 1]
        Charge = f["Charge"][idx : idx + 1]
        collision_energy = f["collision_energy"][idx : idx + 1]
        instrument = f["Fragmentation"][idx : idx + 1]
        instrument = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else s for s in instrument]
        )
        if include_train and "train_data" in f:
            train_data = f["train_data"][idx : idx + 1]
        else:
            train_data = None
    finally:
        if need_close:
            f.close()

    AA = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "[]-": 21,
        "-[]": 22,
        "[Acetyl]-": 38,
        "M[Oxidation]": 23,
        "S[Phospho]": 24,
        "T[Phospho]": 25,
        "Y[Phospho]": 26,
        "K[Dimethyl]": 40,
        "K[Trimethyl]": 41,
        "K[Formyl]": 42,
        "K[Propionyl]": 43,
        "K[Succinyl]": 46,
        "K[Biotin]": 50,
        "K[UNIMOD:737]": 55,
        "R[Dimethyl]": 51,
        "R[UNIMOD:36a]": 52,
        "P[Oxidation]": 53,
        "Y[Nitro]": 54,
        "K[Methyl]": 32,
        "T[HexNAc]": 35,
        "S[HexNAc]": 36,
        "C[Carbamidomethyl]": 37,
        "E[Glu->pyro-Glu]": 39,
        "R[Phospho]": 27,
        "K[Acetyl]": 28,
        "K[GG]": 29,
        "Q[Gln->pyro-Glu]": 30,
        "R[Methyl]": 31,
        "[UNIMOD:737]-": 56,
        "K[UNIMOD:1848]": 47,
        "K[UNIMOD:1363]": 48,
        "K[UNIMOD:1849]": 49,
        "K[UNIMOD:1289]": 44,
        "K[UNIMOD:747]": 45,
    }
    aa_to_idx = {key: idx + 1 for idx, key in enumerate(AA)}

    max_seq_length = SUPPORTED_MAX_LENGTH
    validated_lengths = [validate_length(length) for length in Length]
    Length = torch.tensor(validated_lengths)
    tokens_list = [tokenize_peptide(seq) for seq in Sequence]
    encoded = [[aa_to_idx.get(tok, 0) for tok in tokens] for tokens in tokens_list]
    padded = [
        enc[:max_seq_length] + [0] * (max_seq_length - len(enc))
        if len(enc) < max_seq_length
        else enc[:max_seq_length]
        for enc in encoded
    ]
    sequence_tensor = torch.tensor(padded)
    charge_tensor = torch.tensor([encode_charge(charge) for charge in Charge])
    collision_energy_tensor = torch.tensor(
        [encode_collision_energy(energy) for energy in collision_energy]
    )
    instrument_tensor = torch.tensor(
        [encode_fragmentation(inst) for inst in instrument]
    )
    if train_data is not None:
        train_data = torch.tensor(train_data)

    instrument_tensor = torch.squeeze(instrument_tensor)
    charge_tensor = torch.squeeze(charge_tensor)
    collision_energy_tensor = torch.squeeze(collision_energy_tensor)
    sequence_tensor = torch.squeeze(sequence_tensor)
    Length = torch.squeeze(Length)
    if train_data is not None:
        train_data = torch.squeeze(train_data)
        return (
            instrument_tensor,
            charge_tensor,
            collision_energy_tensor,
            sequence_tensor,
            Length,
            train_data,
        )
    else:
        return (
            instrument_tensor,
            charge_tensor,
            collision_energy_tensor,
            sequence_tensor,
            Length,
        )
