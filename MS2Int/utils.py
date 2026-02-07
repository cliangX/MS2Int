import torch
import numpy as np
try:
    # 该依赖在本项目中并非必需，仅用于部分训练策略场景；
    # 为避免环境未安装 lightning 导致无法运行，这里做兼容处理。
    from lightning.pytorch.strategies import DDPStrategy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DDPStrategy = None

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
    
import torch
import torch.nn.functional as F


def _masked_cosine_core(y_true, y_pred):
    """
    内部工具函数：根据与训练阶段一致的掩码与归一化方式，
    计算每个样本的掩码余弦相似度 cos(theta)。
    """
    # 确保输入是张量
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # 为避免数值不稳定，使用一个很小的浮点数 epsilon
    # 这里使用 Python float，与 CPU/GPU 张量都能正常广播
    epsilon = 1e-7

    # 掩码逻辑：y_true 中为 -1 的位置视为无效位，不参与计算
    # 通过 (y_true + 1) 这一因子，实现对 -1 位置的屏蔽
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 归一化
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)
    true_norm = F.normalize(true_masked, p=2, dim=-1)

    # 计算每个样本的余弦相似度 cos(theta)
    product = torch.sum(pred_norm * true_norm, dim=-1)
    return product


def masked_spectral_distance(y_true, y_pred):
    """
    计算掩码谱图距离（masked spectral distance），与训练阶段使用的定义一致。

    数学形式：
      1. 先在有效位置（非 -1）上计算掩码余弦相似度 cos(theta)
      2. 再将谱图角距离定义为：    d = 2 * arccos(cos(theta)) / pi
    """
    product = _masked_cosine_core(y_true, y_pred)
    arccos = torch.acos(product)
    return 2 * arccos / torch.pi


def masked_cosine_similarity(y_true, y_pred):
    """
    计算与 masked_spectral_distance 使用完全相同掩码与归一化方式的
    掩码余弦相似度（cos(theta)），每个样本返回一个标量。
    """
    return _masked_cosine_core(y_true, y_pred)
#-----------------
# 自定义embedding
# 自定义embedding
#-----------------
import torch
import torch.nn as nn

# 定义嵌入模型
class MetaEmbeddingModel(nn.Module):
    def __init__(self, charge_dim=6, energy_dim=51, instrument_dim=4, final_dim=128):
        super().__init__()
        self.instrument_embedding = nn.Embedding(4, instrument_dim)
        self.charge_embedding = nn.Embedding(6, charge_dim)
        self.collision_energy_embedding = nn.Embedding(12, energy_dim)
        self.aa_embedding = nn.Embedding(25, final_dim, padding_idx=0)
        self.max_aa_length = 30
        self.final_dim = final_dim
        self.fc = nn.Linear(charge_dim + energy_dim + instrument_dim, final_dim)
    
    def forward(self, instrument_idx, charge_idx, collision_energy_idx):
        instrument_embed = self.instrument_embedding(instrument_idx).unsqueeze(1)  # 增加一个维度以便于后续连接
        charge_embed = self.charge_embedding(charge_idx).unsqueeze(1)
        collision_energy_embed = self.collision_energy_embedding(collision_energy_idx).unsqueeze(1)

        concatenated = torch.cat((charge_embed, instrument_embed, collision_energy_embed), dim=2)
        meta_embedding = self.fc(concatenated)
        expanded_meta_embedding = meta_embedding.expand(-1, 30, self.final_dim)
        return expanded_meta_embedding
    

class AminoAcidEmbedding(nn.Module):
    def __init__(self, amino_acid_dim=128):
        super(AminoAcidEmbedding, self).__init__()
        # 假设有20个标准氨基酸，加上几个特殊情况
        self.aa_embedding = nn.Embedding(54, amino_acid_dim)

    def forward(self, aa_idx):
        # Embed amino acids
        aa_emb = self.aa_embedding(aa_idx)
        return aa_emb





import pandas as pd


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
    
    # 计算总参数量占用的显存大小，假设使用 float32，每个参数占用 4 字节
    total_memory_MB = trainable_params * 4 / 1024 / 1024  # 从字节转换到兆字节

    # 表格上半部分
    data = {
        "Name": [model_name],
        "Type": [model_type],
        "Params": [f"{total_params / 1e9:.2f} GParams"],
        "Mode": [mode]
    }
    df_top = pd.DataFrame(data)

    # 表格下半部分
    result_string = f"""
---------------------------------------------------
{df_top.to_string(index=False)}
---------------------------------------------------
{trainable_params / 1e9:.2f} GParams     Trainable params
{non_trainable_params / 1e9:.2f} GParams   Non-trainable params
{total_params / 1e9:.2f} GParams     Total params
{total_memory_MB:.3f} MB   Total estimated model params size (MB)
"""

    return result_string


# -----------------
# 公共函数：损失掩码生成
# -----------------
def create_batch_loss_masks(lengths_list, max_seq_len: int = 29, d_dim: int = 31) -> torch.Tensor:
    """根据每条序列的实际长度生成损失掩码。

    参数：
        lengths_list: 每条序列的长度（可以是张量或列表）
        max_seq_len: 最大序列长度，默认 29
        d_dim: 特征维度，默认 31

    说明：
        - lengths_list 可能是张量或列表，这里统一做 int 转换并裁剪到 [0, max_seq_len]
        - 前 4 列（索引 0..3）在有效位置置 1；后续位置按原逻辑逐步展开
    """
    batch_size = len(lengths_list)
    masks = torch.zeros((batch_size, max_seq_len, d_dim), dtype=torch.int)

    for batch_idx, lengths in enumerate(lengths_list):
        # 兼容张量或 python 数字
        if isinstance(lengths, torch.Tensor):
            L = int(lengths.item())
        else:
            L = int(lengths)
        # 裁剪到合法范围
        L = max(0, min(L, max_seq_len))

        # 前 4 列直到当前长度设置为 1（排除最后一个位置与原逻辑一致）
        if L > 0:
            masks[batch_idx, : max(L - 1, 0), :4] = 1

        # 从长度 L 开始，后面的列的前几列逐步增加 1
        for idx, lens in enumerate(range(L, 3, -1)):
            if idx < max_seq_len:
                masks[batch_idx, idx, 3 : min(lens + 1, d_dim)] = 1

    return masks


# -----------------
# 公共函数：加载检查点
# -----------------
def load_checkpoint(checkpoint_path: str, model: torch.nn.Module):
    """加载模型检查点。

    参数：
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型

    返回：
        (epoch, val_loss): 检查点保存时的 epoch 和验证损失
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict)
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']

    print(f"Loaded checkpoint '{checkpoint_path}' from epoch {epoch} with validation loss {val_loss:.4f}.")
    return epoch, val_loss
