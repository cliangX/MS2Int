import atexit
import argparse
import gc
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
parser.add_argument(
    "--predict_key",
    "--dataset_name",
    dest="predict_key",
    default="Intpredict",
    help="Prediction dataset name written to output HDF5",
)
parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")
parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite prediction dataset when it already exists",
)
args = parser.parse_args()

REQUIRED_COLS = ["Sequence", "Length", "Charge", "collision_energy", "Fragmentation"]


def _csv_to_h5(csv_path: str, h5_path: str) -> None:
    sep = "\t" if os.path.splitext(csv_path)[1].lower() == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV/TSV missing required columns: {missing}")

    if "annotate" in df.columns:
        annotate_col = df["annotate"].astype(str)
    elif "Modified_sequence" in df.columns:
        annotate_col = df["Modified_sequence"].astype(str)
    else:
        annotate_col = df["Sequence"].astype(str)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "Sequence",
            data=df["Sequence"].astype(str).str.encode("utf-8").values.astype("S128"),
        )
        f.create_dataset(
            "annotate",
            data=annotate_col.str.encode("utf-8").values.astype("S256"),
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
predict_key = args.predict_key

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

batch_size = args.batch_size
num_workers = args.num_workers

# 不覆盖外部传入的 CUDA_VISIBLE_DEVICES；例如 CUDA_VISIBLE_DEVICES=6 时，
# 进程内的 cuda:0 会映射到物理 GPU6。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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


def test(model, test_loader, device, valid_indices, total_samples, prediction_path):
    model.eval()
    test_progress = tqdm(test_loader, ncols=30, desc="Testing")
    pred_ds = None
    written = 0

    with h5py.File(prediction_path, "w") as pred_file:
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
                outputs_np = outputs.cpu().numpy().astype(np.float32, copy=False)

                if pred_ds is None:
                    pred_shape = (total_samples,) + tuple(outputs_np.shape[1:])
                    chunk_rows = min(batch_size, total_samples)
                    pred_ds = pred_file.create_dataset(
                        predict_key,
                        shape=pred_shape,
                        dtype=np.float32,
                        chunks=(chunk_rows,) + tuple(outputs_np.shape[1:]),
                    )

                batch_rows = valid_indices[written : written + outputs_np.shape[0]]
                pred_ds[batch_rows] = outputs_np
                written += outputs_np.shape[0]

    print(f"Completed testing, total samples: {written}")


def _copy_prediction_to_output(
    input_path: str,
    output_path: str,
    prediction_path: str,
    predict_key: str,
    overwrite: bool,
) -> None:
    same_file = os.path.abspath(output_path) == os.path.abspath(input_path)
    output_exists = os.path.exists(output_path)

    if same_file or output_exists:
        with h5py.File(output_path, "a") as output_file, h5py.File(
            prediction_path, "r"
        ) as pred_file:
            if predict_key in output_file:
                if not overwrite:
                    raise KeyError(
                        f"输出 H5 已存在数据集 {predict_key!r}；如需覆盖请添加 --overwrite"
                    )
                del output_file[predict_key]
            pred_file.copy(predict_key, output_file, name=predict_key)
        return

    with (
        h5py.File(input_path, "r") as input_file,
        h5py.File(output_path, "w") as output_file,
        h5py.File(prediction_path, "r") as pred_file,
    ):
        for attr_key, attr_value in input_file.attrs.items():
            output_file.attrs[attr_key] = attr_value
        for key in input_file.keys():
            # 输入里可能已有与 predict_key 同名的旧预测；应用新预测前不要复制该键
            if key == predict_key:
                continue
            input_file.copy(key, output_file, name=key)
        pred_file.copy(predict_key, output_file, name=predict_key)


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

valid_indices = np.asarray(valid_indices, dtype=np.int64)
prediction_tmp = tempfile.NamedTemporaryFile(
    suffix=f".{predict_key}.h5",
    prefix="ms2int_pred_",
    dir=output_dir if output_dir else None,
    delete=False,
)
prediction_tmp.close()
try:
    test(model, test_loader, device, valid_indices, total_samples, prediction_tmp.name)
    del test_loader, filtered_dataset, original_dataset
    gc.collect()
    _copy_prediction_to_output(
        input_path=input_path,
        output_path=output_path,
        prediction_path=prediction_tmp.name,
        predict_key=predict_key,
        overwrite=args.overwrite,
    )
finally:
    if os.path.exists(prediction_tmp.name):
        os.remove(prediction_tmp.name)
