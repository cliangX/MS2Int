import atexit
import argparse
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mamba_ssm.models.config_mamba import MambaConfig

try:
    from .model import MambaLMHeadModel
    from .datasets import CustomDataset
    from .utils import *
    from .metadata_vocab import SUPPORTED_MAX_LENGTH
except ImportError:  # pragma: no cover
    from model import MambaLMHeadModel
    from datasets import CustomDataset
    from utils import *
    from metadata_vocab import SUPPORTED_MAX_LENGTH

parser = argparse.ArgumentParser(description="Spectrum prediction")
parser.add_argument(
    "--checkpoint_path",
    "--ckpt",
    dest="checkpoint_path",
    required=True,
    help="Model checkpoint path (.pth)",
)
parser.add_argument(
    "--input_path",
    "--input",
    dest="input_path",
    required=True,
    help="Input file path (.h5, .csv, or .tsv)",
)
parser.add_argument(
    "--output_path",
    "--output",
    dest="output_path",
    required=True,
    help="Output HDF5 path",
)
args = parser.parse_args()

REQUIRED_COLS = ["Sequence", "Length", "Charge", "collision_energy", "Fragmentation"]


def _csv_to_h5(csv_path: str, h5_path: str) -> None:
    sep = "\t" if os.path.splitext(csv_path)[1].lower() == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV/TSV missing required columns: {missing}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "Sequence",
            data=df["Sequence"].astype(str).str.encode("utf-8").values.astype("S128"),
        )
        f.create_dataset("Length", data=df["Length"].values.astype(np.int32))
        f.create_dataset("Charge", data=df["Charge"].values.astype(np.int32))
        f.create_dataset(
            "collision_energy", data=df["collision_energy"].values.astype(np.int32)
        )
        f.create_dataset(
            "Fragmentation",
            data=df["Fragmentation"]
            .astype(str)
            .str.encode("utf-8")
            .values.astype("S10"),
        )
    print(f"Converted {csv_path} -> {h5_path} ({len(df)} samples)")


checkpoint_path = args.checkpoint_path
raw_input_path = args.input_path
output_path = args.output_path

_tmp_h5 = None
ext = os.path.splitext(raw_input_path)[1].lower()
if ext in (".csv", ".tsv"):
    _tmp_h5 = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    _tmp_h5.close()
    _csv_to_h5(raw_input_path, _tmp_h5.name)
    input_path = _tmp_h5.name
    atexit.register(lambda: os.path.exists(_tmp_h5.name) and os.remove(_tmp_h5.name))
else:
    input_path = raw_input_path

batch_size = 1024
num_workers = 8
gpu_id = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

Mamba_Config = MambaConfig(
    d_model=512,
    d_intermediate=0,
    n_layer=4,
    ssm_cfg={"layer": "Mamba2"},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    tie_embeddings=True,
)

device = "cuda:0"

model = MambaLMHeadModel(Mamba_Config)
epoch, val_loss = load_checkpoint(checkpoint_path, model)
model = model.to(device)


def test(model, test_loader, device):
    model.eval()
    all_y_outputs = []
    test_progress = tqdm(test_loader, ncols=30, desc="Testing")

    with torch.inference_mode():
        for batch in test_progress:
            inst, charge, ce, seq, lengths = batch
            inst = inst.to(device, non_blocking=True)
            charge = charge.to(device, non_blocking=True)
            ce = ce.to(device, non_blocking=True)
            seq = seq.to(device, non_blocking=True)

            outputs = model(inst, charge, ce, seq)
            masks = create_batch_loss_masks(lengths.tolist()).to(
                device, non_blocking=True
            )
            outputs[outputs < 0] = 0
            outputs = outputs * masks
            all_y_outputs.append(outputs.cpu())

    all_y_outputs = torch.cat(all_y_outputs, dim=0)
    print(f"Completed testing, total samples: {all_y_outputs.shape[0]}")
    return all_y_outputs


output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with h5py.File(input_path, "r") as f:
    length_ds = f["Length"]
    seq_ds = f.get("Sequence", None)
    total_samples = int(length_ds.shape[0])

    # Unified filter: Length<=40 and sequence must not contain 'U'
    chunk_size = 200_000
    valid_indices = []
    filtered_by_length = 0
    filtered_by_u = 0

    if seq_ds is None:
        print(
            f"Warning: input H5 is missing the Sequence dataset; skipping U-filter (Length<={SUPPORTED_MAX_LENGTH} filter only)"
        )

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        lengths = np.asarray(length_ds[start:end])
        is_len_ok = lengths <= SUPPORTED_MAX_LENGTH

        if seq_ds is None:
            no_u = np.ones_like(is_len_ok, dtype=bool)
        else:
            seq_chunk = np.asarray(seq_ds[start:end])
            if seq_chunk.dtype.kind == "O":
                # Compat: convert variable-length str/bytes to fixed-length bytes for vectorized search
                seq_chunk = seq_chunk.astype("S")

            if seq_chunk.dtype.kind in ("S", "a"):
                no_u = np.char.find(seq_chunk, b"U") == -1
            else:
                no_u = np.char.find(seq_chunk, "U") == -1

        is_ok = is_len_ok & no_u
        filtered_by_length += int((~is_len_ok).sum())
        filtered_by_u += int((is_len_ok & ~no_u).sum())
        valid_indices.extend((np.nonzero(is_ok)[0] + start).tolist())

    print(
        f"Samples: {len(valid_indices)}/{total_samples} "
        f"(filtered length>{SUPPORTED_MAX_LENGTH}: {filtered_by_length}, filtered U: {filtered_by_u})"
    )

original_dataset = CustomDataset(input_path, include_train=False)
filtered_dataset = Subset(original_dataset, valid_indices)
test_loader = DataLoader(
    filtered_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
)

all_y_outputs = test(model, test_loader, device)

with h5py.File(input_path, "r") as f_in:
    total_samples = f_in["Length"].shape[0]

full_pred = np.zeros(
    (total_samples,) + tuple(all_y_outputs.shape[1:]), dtype=np.float32
)
full_pred[valid_indices] = all_y_outputs.numpy()

if os.path.abspath(output_path) == os.path.abspath(input_path):
    with h5py.File(input_path, "a") as f:
        if "Intpredict" in f:
            del f["Intpredict"]
        f.create_dataset("Intpredict", data=full_pred)
else:
    with (
        h5py.File(input_path, "r") as input_file,
        h5py.File(output_path, "w") as output_file,
    ):
        for key in input_file.keys():
            data = input_file[key][:]
            output_file.create_dataset(key, data=data)

            if "description" in input_file[key].attrs:
                output_file[key].attrs["description"] = input_file[key].attrs[
                    "description"
                ]

        output_file.create_dataset("Intpredict", data=full_pred)
