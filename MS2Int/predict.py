# %%
# 用户自定义参数配置
# ==================================================
# GPU设置
"""
python predict.py \
  --ckpt "checkpoints/model.pth" \
  --input "data/input.h5" \
  --output "outputs/pred.h5"
"""

gpu_id = "0"  # 使用的GPU ID，例如："0"、"1"或"0,1"表示使用多个GPU


# 模型路径
# 模型路径通过命令行传参
import argparse

parser = argparse.ArgumentParser(description="Spectrum prediction")
parser.add_argument(
    "--checkpoint_path",
    "--ckpt",
    dest="checkpoint_path",
    required=True,
    help="模型检查点路径(.pth)",
)
parser.add_argument(
    "--input_path", "--input", dest="input_path", required=True, help="输入数据HDF5路径"
)
parser.add_argument(
    "--output_path",
    "--output",
    dest="output_path",
    required=True,
    help="输出结果HDF5路径",
)
args = parser.parse_args()
checkpoint_path = args.checkpoint_path
input_path = args.input_path
output_path = args.output_path
# 批处理大小
batch_size = 1024

# 其他参数
num_workers = 8  # 数据加载线程数
# ==================================================

# %%
import torch

print(torch.cuda.is_available())

# %%
import os

# 设置当前使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# %%
# 标准库
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from datasets import CustomDataset

# 这里保持原有导入不变
from mamba_ssm.models.config_mamba import MambaConfig

# 不再需要添加模型路径，因为已经将文件复制到当前目录
from model import MambaLMHeadModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils import *

# %%
Mamba_Config = MambaConfig(
    d_model=512,
    d_intermediate=0,
    n_layer=4,  # 修改为训练时的层数
    ssm_cfg={"layer": "Mamba2"},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    tie_embeddings=True,
)

# %%
device = "cuda:0"  # 直接使用GPU


# load_checkpoint 已移至 utils.py

# 用相同的模型架构初始化你的模型
model = MambaLMHeadModel(Mamba_Config)
epoch, val_loss = load_checkpoint(checkpoint_path, model)
model = model.to(device)


# create_batch_loss_masks 和 masked_spectral_distance 已移至 utils.py


def test(model, test_loader, device):
    model.eval()
    all_y_outputs = []
    test_progress = tqdm(test_loader, ncols=30, desc="Testing")

    with torch.inference_mode():
        for batch in test_progress:
            # 只将必要张量搬到 GPU，lengths 留在 CPU
            inst, charge, ce, seq, lengths = batch
            inst = inst.to(device, non_blocking=True)
            charge = charge.to(device, non_blocking=True)
            ce = ce.to(device, non_blocking=True)
            seq = seq.to(device, non_blocking=True)

            outputs = model(inst, charge, ce, seq)
            # 使用 CPU 端 lengths 构造掩码，再搬到 GPU
            masks = create_batch_loss_masks(lengths.tolist()).to(
                device, non_blocking=True
            )
            outputs[outputs < 0] = 0
            outputs = outputs * masks

            all_y_outputs.append(outputs.cpu())

    all_y_outputs = torch.cat(all_y_outputs, dim=0)
    print(f"Completed testing, total samples: {all_y_outputs.shape[0]}")
    return all_y_outputs


# %%
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 添加的肽段长度过滤代码
print("正在读取肽段长度信息...")
with h5py.File(input_path, "r") as f:
    lengths = f["Length"][:]
    total_samples = len(lengths)

    # 创建符合条件的样本索引列表（长度<=30的肽段）
    valid_indices = [i for i in range(total_samples) if lengths[i] <= 30]
    filtered_count = total_samples - len(valid_indices)

    print(f"总样本数: {total_samples}")
    print(
        f"长度>30的样本数: {filtered_count} ({filtered_count / total_samples * 100:.2f}%)"
    )
    print(f"过滤后的样本数: {len(valid_indices)}")

# 实例化原始数据集（推理阶段不需要 train_data，避免额外 I/O）
original_dataset = CustomDataset(input_path, include_train=False)

# 创建过滤后的子集
filtered_dataset = Subset(original_dataset, valid_indices)

# 用过滤后的数据集创建DataLoader
test_loader = DataLoader(
    filtered_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
)

# %%
# 进行推理
print("开始进行模型推理...")
all_y_outputs = test(model, test_loader, device)

# %%
# 保存结果 - 不仅保存预测结果，还保留原始数据的所有键
# 修改这里：对所有字段都应用相同的过滤
print("保存结果到h5文件...")
# 构建与原始样本数一致的全长预测矩阵，未参与预测的位置填 0
with h5py.File(input_path, "r") as f_in:
    total_samples = f_in["Length"].shape[0]

full_pred = np.zeros(
    (total_samples,) + tuple(all_y_outputs.shape[1:]), dtype=np.float32
)
full_pred[valid_indices] = all_y_outputs.numpy()

# 若输出路径与输入路径相同，则直接在原文件中追加/覆盖 Intpredict 数据集
if os.path.abspath(output_path) == os.path.abspath(input_path):
    print("输出路径与输入路径相同，将在原 H5 文件中追加 Intpredict 数据集。")
    with h5py.File(input_path, "a") as f:
        if "Intpredict" in f:
            del f["Intpredict"]
        f.create_dataset("Intpredict", data=full_pred)
else:
    # 否则保持原有行为：新建输出文件，复制所有原始字段，再写入 Intpredict
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
    # output_file.create_dataset('Intpredict_loss', data=test_losses)
# %%
# 删除输入文件 - 已禁用，保留输入文件以便重复运行
# if os.path.exists(input_path):
#     os.remove(input_path)


# %%
# 查看结果文件结构
# print("\n结果文件结构:")
# with h5py.File(output_path, 'r') as f:
#
# ("文件中的键:", list(f.keys()))
# for key in f.keys():
# dataset = f[key]
# print(f"键: {key}, 形状: {dataset.shape}, 数据类型: {dataset.dtype}")
# if 'description' in dataset.attrs:
# print(f"  描述: {dataset.attrs['description']}")

# # %%
# 查看第一条数据
# print("\n第一条数据信息:")
# with h5py.File(output_path, 'r') as f:
# for key in f.keys():
# try:
# if key == 'train_data' or key == 'Intpredict':
# print(f"{key}: 形状为 {f[key][0].shape}")
# elif f[key].dtype.kind == 'S' or f[key].dtype.kind == 'O':
# try:
# value = f[key][0]
# if isinstance(value, bytes):
# print(f"{key}: {value.decode('utf-8')}")
# else:
# print(f"{key}: {value}")
# except:
# print(f"{key}: [无法解码]")
# else:
# print(f"{key}: {f[key][0]}")
# except Exception as e:
# print(f"{key}: 无法显示 - {str(e)}")


# 读取loss数据
# with h5py.File(output_path, 'r') as f:
# if 'Intpredict_loss' in f:
# losses = f['Intpredict_loss'][:]

#         # 计算统计数据
# print(f"Loss统计信息:")
# print(f"样本数量: {len(losses)}")
# print(f"平均值: {np.mean(losses):.4f}")
# print(f"中位数: {np.median(losses):.4f}")
# print(f"标准差: {np.std(losses):.4f}")
# print(f"最小值: {np.min(losses):.4f}")
# print(f"最大值: {np.max(losses):.4f}")

#         # 计算分位数
# percentiles = [10, 25, 50, 75, 90, 95, 99]
# print("\nLoss分位数:")
# for p in percentiles:
# value = np.percentile(losses, p)
# print(f"{p}%: {value:.4f}")

#         # 统计不同范围内的loss分布
# ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0), (1.0, float('inf'))]
# print("\nLoss分布:")
# for low, high in ranges:
# count = np.sum((losses >= low) & (losses < high))
# percentage = count / len(losses) * 100
# print(f"{low:.1f} - {high if high != float('inf') else '∞'}: {count} 个样本 ({percentage:.2f}%)")
# else:
# print("找不到'Intpredict_loss'数据集")

#
#
