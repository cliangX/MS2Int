import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

try:
    from .metadata_vocab import (
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_FRAGMENTATIONS,
        SUPPORTED_MAX_LENGTH,
        TARGET_OUTPUT_SHAPE,
    )
except ImportError:  # pragma: no cover
    from metadata_vocab import (
        SUPPORTED_CHARGES,
        SUPPORTED_COLLISION_ENERGIES,
        SUPPORTED_FRAGMENTATIONS,
        SUPPORTED_MAX_LENGTH,
        TARGET_OUTPUT_SHAPE,
    )


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def _masked_cosine_core(y_true, y_pred):
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    epsilon = 1e-7
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    pred_norm = F.normalize(pred_masked, p=2, dim=-1)
    true_norm = F.normalize(true_masked, p=2, dim=-1)

    product = torch.sum(pred_norm * true_norm, dim=-1)
    return product


def masked_spectral_distance(y_true, y_pred):
    product = _masked_cosine_core(y_true, y_pred)
    return 2 * torch.acos(product) / torch.pi


def masked_cosine_similarity(y_true, y_pred):
    return _masked_cosine_core(y_true, y_pred)


class MetaEmbeddingModel(nn.Module):
    def __init__(self, charge_dim=6, energy_dim=51, instrument_dim=4, final_dim=128):
        super().__init__()
        self.instrument_embedding = nn.Embedding(
            len(SUPPORTED_FRAGMENTATIONS), instrument_dim
        )
        self.charge_embedding = nn.Embedding(len(SUPPORTED_CHARGES), charge_dim)
        self.collision_energy_embedding = nn.Embedding(
            len(SUPPORTED_COLLISION_ENERGIES), energy_dim
        )
        self.aa_embedding = nn.Embedding(25, final_dim, padding_idx=0)
        self.max_aa_length = SUPPORTED_MAX_LENGTH
        self.final_dim = final_dim
        self.fc = nn.Linear(charge_dim + energy_dim + instrument_dim, final_dim)

    def forward(self, instrument_idx, charge_idx, collision_energy_idx):
        instrument_embed = self.instrument_embedding(instrument_idx).unsqueeze(1)
        charge_embed = self.charge_embedding(charge_idx).unsqueeze(1)
        collision_energy_embed = self.collision_energy_embedding(
            collision_energy_idx
        ).unsqueeze(1)
        concatenated = torch.cat(
            (charge_embed, instrument_embed, collision_energy_embed), dim=2
        )
        meta_embedding = self.fc(concatenated)
        return meta_embedding.expand(-1, self.max_aa_length, self.final_dim)


class AminoAcidEmbedding(nn.Module):
    def __init__(self, amino_acid_dim=128):
        super().__init__()
        self.aa_embedding = nn.Embedding(54, amino_acid_dim)

    def forward(self, aa_idx):
        return self.aa_embedding(aa_idx)


def count_parameters(model, model_name="model", model_type="Mamba2Gate", mode="train"):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()

    total_memory_MB = trainable_params * 4 / 1024 / 1024

    data = {
        "Name": [model_name],
        "Type": [model_type],
        "Params": [f"{total_params / 1e9:.2f} GParams"],
        "Mode": [mode],
    }
    df_top = pd.DataFrame(data)

    return f"""
---------------------------------------------------
{df_top.to_string(index=False)}
---------------------------------------------------
{trainable_params / 1e9:.2f} GParams     Trainable params
{non_trainable_params / 1e9:.2f} GParams   Non-trainable params
{total_params / 1e9:.2f} GParams     Total params
{total_memory_MB:.3f} MB   Total estimated model params size (MB)
"""


def create_batch_loss_masks(
    lengths_list,
    max_seq_len: int = TARGET_OUTPUT_SHAPE[0],
    d_dim: int = TARGET_OUTPUT_SHAPE[1],
) -> torch.Tensor:
    batch_size = len(lengths_list)
    masks = torch.zeros((batch_size, max_seq_len, d_dim), dtype=torch.int)

    for batch_idx, lengths in enumerate(lengths_list):
        if isinstance(lengths, torch.Tensor):
            L = int(lengths.item())
        else:
            L = int(lengths)
        L = max(0, min(L - 1, max_seq_len))

        if L > 0:
            masks[batch_idx, :L, :4] = 1

        for idx, lens in enumerate(range(L + 1, 3, -1)):
            if idx < max_seq_len:
                masks[batch_idx, idx, 3 : min(lens + 1, d_dim)] = 1

    return masks


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    new_state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
    }
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint 与当前 40aa/39x41 训练契约不兼容，请使用新契约训练得到的权重，"
            f"或留空 --pth 重新开始 fine-tune。原始错误: {exc}"
        ) from exc
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    print(
        f"Loaded checkpoint '{checkpoint_path}' from epoch {epoch} with validation loss {val_loss:.4f}."
    )
    return epoch, val_loss
