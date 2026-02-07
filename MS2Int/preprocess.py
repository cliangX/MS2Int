import h5py
import torch
import numpy as np

def tokenize_peptide(seq: str):
    tokens = []
    i, n = 0, len(seq)
    if seq.startswith('['):
        j = seq.find(']-', 0)
        if j == -1:
            raise ValueError("N-terminus modification missing ']-'")
        tokens.append(seq[:j + 2])
        i = j + 2
    while i < n:
        if seq.startswith('-[]', i):
            tokens.append('-[]')
            i += 3
            continue
        ch = seq[i]
        if 'A' <= ch <= 'Z':
            i += 1
            if i < n and seq[i] == '[':
                j = seq.find(']', i)
                if j == -1:
                    raise ValueError("Residue modification missing ']' ")
                tokens.append(ch + seq[i:j + 1])
                i = j + 1
            else:
                tokens.append(ch)
        else:
            raise ValueError(f"Invalid character '{ch}' at pos {i}")
    return tokens

def data_read(h5_or_path, idx, include_train: bool = True):
    """Read one sample from HDF5 and convert to tensors.

    If h5_or_path is a string, open/close the file for this call.
    If it's an h5py.File handle, reuse it (no open/close here).

    When include_train=False, skip reading 'train_data' to save I/O for inference.
    """
    need_close = False
    if isinstance(h5_or_path, str):
        f = h5py.File(h5_or_path, 'r')
        need_close = True
    else:
        f = h5_or_path

    try:
        # 优先使用 ProForma 风格的 annotate 序列（包含 [Phospho] 等修饰）
        if "annotate" in f:
            Sequence = f["annotate"][idx : idx + 1]
        else:
            Sequence = f["Sequence"][idx : idx + 1]
        Sequence = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else s for s in Sequence]
        )
        Length = f['Length'][idx:idx+1]
        Charge = f['Charge'][idx:idx+1]
        collision_energy = f['collision_energy'][idx:idx+1]
        instrument = f["Fragmentation"][idx : idx + 1]
        instrument = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in instrument])
        if include_train and 'train_data' in f:
            train_data = f['train_data'][idx:idx+1]
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
    instruments = ["HCD", "CID"]
    charges = list(range(1, 7))  # 1 到 6
    collision_energies = list([10,20,23,25,26,27,28,29,30,35,40,42])  # 10 到 50

    # 为每种类型创建字典，映射每个值到一个索引
    instrument_to_idx = {inst: idx for idx, inst in enumerate(instruments)}
    charge_to_idx = {charge: idx for idx, charge in enumerate(charges)}
    collision_energy_to_idx = {ce: idx for idx, ce in enumerate(collision_energies)}
    # 创建氨基酸序列索引映射
    aa_to_idx = {key: idx + 1 for idx, key in enumerate(AA)}  # +1 因为0是填充索引


    # 使用前面定义的索引字典将输入转换为Tensor
    max_seq_length = 30  # 定义最大序列长度
    Length = torch.tensor(Length)
    tokens_list = [tokenize_peptide(seq) for seq in Sequence]
    encoded = [[aa_to_idx.get(tok, 0) for tok in tokens] for tokens in tokens_list]
    padded = [enc[:max_seq_length] + [0] * (max_seq_length - len(enc)) if len(enc) < max_seq_length else enc[:max_seq_length] for enc in encoded]
    sequence_tensor = torch.tensor(padded)
    charge_tensor = torch.tensor([charge_to_idx.get(charge, 0) for charge in Charge])
    collision_energy_tensor = torch.tensor([collision_energy_to_idx.get(int(energy), 0) for energy in collision_energy])
    instrument_tensor = torch.tensor([instrument_to_idx.get(inst, 0) for inst in instrument])
    if train_data is not None:
        train_data = torch.tensor(train_data)

    # 创建0和-1的张量，它们的形状与data在除了最后一个维度以外的维度相同
    # zeros = torch.zeros(train_data.size(0), 1, train_data.size(2))  # 在最后一个维度添加1列0
    # neg_ones = torch.full((train_data.size(0), 1, train_data.size(2)), 0)  # 在最后一个维度添加1列0

    # 在第三维度最前面加一串0
    #src_train_data = train_data

    # 在第三维度最后面加一串-1

    instrument_tensor = torch.squeeze(instrument_tensor)
    charge_tensor = torch.squeeze(charge_tensor)
    collision_energy_tensor = torch.squeeze(collision_energy_tensor)
    sequence_tensor = torch.squeeze(sequence_tensor)
    Length = torch.squeeze(Length)
    if train_data is not None:
        train_data = torch.squeeze(train_data)
        return instrument_tensor, charge_tensor, collision_energy_tensor, sequence_tensor, Length, train_data
    else:
        return instrument_tensor, charge_tensor, collision_energy_tensor, sequence_tensor, Length
