"""不带 VAT 的微调脚本（与 fine_tune.py 平行，仅去掉 VAT 损失）。

要点
- 在 --pth 给定的预训练 ckpt 上加载权重；
- 冻结浅层：embedding / 前 (n_layer - K) 个 forward/backward block / 对应 hidden_fc；
- 解冻：backbone 最后 K 层 forward/backward block + hidden_fc[i] + gate + norm_f + lm_head；
- 监督损失同 main.validate：masked_spectral_distance（合法零峰保留为 0，仅由结构 mask 排除不存在位置）；
- 不引入对抗扰动，相对小数据更稳定，避免 VAT 在 1.6k 量级的过强约束；
- 数据契约与 metadata_vocab.SUPPORTED_* 对齐（40aa, 39x41, 7 个 CE bin）。
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mamba_ssm.models.config_mamba import MambaConfig

try:
    from .model import MambaLMHeadModel
    from .utils import (
        CosineWarmupScheduler,
        count_parameters,
        create_batch_loss_masks,
        load_checkpoint,
        masked_spectral_distance,
    )
    from .datasets import CustomDataset
except ImportError:  # pragma: no cover
    from model import MambaLMHeadModel
    from utils import (
        CosineWarmupScheduler,
        count_parameters,
        create_batch_loss_masks,
        load_checkpoint,
        masked_spectral_distance,
    )
    from datasets import CustomDataset

try:
    from lightning.pytorch import seed_everything
except ModuleNotFoundError:  # pragma: no cover

    def seed_everything(seed: int = 42, workers: bool = True) -> None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================== 参数 ==============================
def get_hparams():
    p = argparse.ArgumentParser(
        description="MS2Int fine-tune (no VAT, with frozen shallow layers)"
    )
    p.add_argument("--experiment_name", type=str, default="MS2Int_FT_noVAT")
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument(
        "--pth",
        type=str,
        required=True,
        help="预训练 ckpt 路径（必填）",
    )
    p.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="混合训练 H5 路径（mix_train.h5）",
    )
    p.add_argument(
        "--val_data_path",
        type=str,
        default="",
        help="独立验证集 H5；为空则从 train 按 train_data_size 切分（同 fine_tune.py）",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="ckpt 输出目录",
    )
    p.add_argument("--log_path", type=str, default="logs/ft_novat.log")

    # 数据相关
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--train_data_size",
        type=float,
        default=0.99,
        help="无 --val_data_path 时按此比例从 train 切训练集（剩余作 val）",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="将整个 H5 一次性加载到内存（mix_train.h5 ~22 GB，需 ≥32 GB 空闲内存）",
    )

    # 优化器/调度
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--max_iters", type=int, default=200_000)
    p.add_argument("--max_epochs", type=int, default=8)

    # 模型结构（必须与 ckpt 匹配）
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_model", type=int, default=512)

    # 冻结策略
    p.add_argument(
        "--freeze_last_k",
        type=int,
        default=1,
        help="解冻 backbone 最后 K 层 forward/backward block + hidden_fc[i]；"
        "其余 backbone 层、embedding、expansion_layer、transformation_matrix 全部冻结。"
        "lm_head / gate / norm_f 始终解冻。",
    )
    p.add_argument(
        "--save_every_epoch",
        action="store_true",
        default=False,
        help="每个 epoch 都保存 ckpt（默认仅在 val_loss 改善时保存）",
    )

    # DDP
    p.add_argument("--server", type=str, default="localhost")
    p.add_argument("--port", type=str, default="29500")
    return p.parse_args()


config = get_hparams()

try:
    import setproctitle  # type: ignore

    setproctitle.setproctitle(config.experiment_name)
except ModuleNotFoundError:  # pragma: no cover
    pass


log_dir = os.path.dirname(config.log_path)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
if config.checkpoint_path:
    os.makedirs(config.checkpoint_path, exist_ok=True)

logging.basicConfig(
    filename=config.log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)


Mamba_Config = MambaConfig(
    d_model=config.d_model,
    d_intermediate=0,
    n_layer=config.n_layer,
    ssm_cfg={"layer": "Mamba2"},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    tie_embeddings=True,
)


# ============================== 工具 ==============================
def _set_finetune_freeze(m: torch.nn.Module, last_k: int = 1):
    """冻结 embedding / 前若干层；
    解冻：最后 last_k 层 forward/backward block + hidden_fc[i] + gate + norm_f + lm_head。

    返回 (n_total, n_train) 用于打日志。
    """
    for p in m.parameters():
        p.requires_grad = False

    bb = m.backbone
    n = len(bb.forward_layers)
    last_k = max(1, min(last_k, n))
    train_idx = set(range(n - last_k, n))

    for i in train_idx:
        for p in bb.forward_layers[i].parameters():
            p.requires_grad = True
        for p in bb.backward_layers[i].parameters():
            p.requires_grad = True
        for p in bb.hidden_fc[i].parameters():
            p.requires_grad = True

    for p in bb.gate.parameters():
        p.requires_grad = True
    for p in bb.norm_f.parameters():
        p.requires_grad = True
    for p in m.lm_head.parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return n_total, n_train, sorted(train_idx)


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _resolve_master_port(preferred_port) -> int:
    try:
        port = int(preferred_port)
    except (TypeError, ValueError):
        port = 0
    if port > 0 and _is_port_available(port):
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = str(config.server)
    os.environ["MASTER_PORT"] = str(os.environ.get("MS2INT_MASTER_PORT", config.port))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# ============================== 训练 ==============================
def train(rank, world_size, mamba_cfg):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = MambaLMHeadModel(mamba_cfg)
    epoch_loaded, val_loss_loaded = load_checkpoint(config.pth, model)
    n_total, n_train, train_idx = _set_finetune_freeze(
        model, last_k=config.freeze_last_k
    )

    if rank == 0:
        logging.info("=" * 60)
        logging.info(
            f"启动 fine-tune (no VAT)  ckpt={config.pth}  "
            f"loaded epoch={epoch_loaded}  val_loss={val_loss_loaded:.4f}"
        )
        logging.info(
            f"参数总量={n_total/1e6:.2f}M  可训={n_train/1e6:.2f}M  "
            f"占比={n_train/max(n_total,1)*100:.1f}%"
        )
        logging.info(
            f"冻结策略: last_k={config.freeze_last_k}, 解冻 block 索引={train_idx}, "
            f"始终解冻 lm_head / gate / norm_f"
        )
        logging.info(
            f"训练数据={config.train_data_path}  "
            f"val_data={'(从 train 切)' if not config.val_data_path else config.val_data_path}"
        )
        logging.info(
            f"lr={config.learning_rate}  weight_decay={config.weight_decay}  "
            f"warmup={config.warmup_iters}  max_iters={config.max_iters}  "
            f"max_epochs={config.max_epochs}"
        )
        logging.info(count_parameters(model))

    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    train_dataset = CustomDataset(config.train_data_path, preload=config.preload)
    if config.val_data_path:
        val_dataset = CustomDataset(config.val_data_path, preload=config.preload)
        if rank == 0:
            logging.info(
                f"独立验证集: {config.val_data_path} ({len(val_dataset)} 样本)"
            )
    else:
        N = len(train_dataset)
        n_train_split = int(config.train_data_size * N)
        n_val_split = N - n_train_split
        gen = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            train_dataset, [n_train_split, n_val_split], generator=gen
        )
        if rank == 0:
            logging.info(
                f"按比例切分: train={n_train_split}  val={n_val_split} (seed=42)"
            )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup=config.warmup_iters, max_iters=config.max_iters
    )

    best_val = 1e9
    for epoch in range(config.max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        if rank == 0:
            pbar = tqdm(
                train_loader,
                ncols=100,
                desc=f"Epoch {epoch+1}/{config.max_epochs} FT-noVAT",
            )
        else:
            pbar = train_loader

        for batch in pbar:
            batch = [t.to(device, non_blocking=True) for t in batch]
            optimizer.zero_grad()
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            masks = create_batch_loss_masks(batch[4]).to(device, non_blocking=True)

            outputs_sup = outputs * masks
            tgt = batch[-1] * masks
            y_p = outputs_sup.reshape(outputs_sup.size(0), -1)
            y_t = tgt.reshape(tgt.size(0), -1)
            loss = masked_spectral_distance(y_t, y_p).sum() / batch[0].size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            li = float(loss.item())
            epoch_loss += li
            if rank == 0:
                pbar.set_postfix({"train_loss": li})

        loss_t = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        avg = loss_t.item() / (world_size * len(train_loader))
        if rank == 0:
            logging.info(f"epoch {epoch} avg train loss: {avg:.4f}")

        val = validate(rank, world_size, model, val_loader, device, epoch)

        if rank == 0:
            improved = val < best_val
            if improved:
                best_val = val
            if improved or config.save_every_epoch:
                save_ckpt(rank, model, optimizer, epoch, val, config.checkpoint_path)
            logging.info(
                f"epoch {epoch} END  train_loss={avg:.4f}  val_loss={val:.4f}  "
                f"best_val={best_val:.4f}  improved={improved}"
            )

    cleanup()


def validate(rank, world_size, model, val_loader, device, epoch):
    model.eval()
    val_sum = 0.0

    if rank == 0:
        pbar = tqdm(val_loader, ncols=100, desc=f"Val   epoch {epoch+1}")
    else:
        pbar = val_loader

    with torch.no_grad():
        for batch in pbar:
            batch = [t.to(device) for t in batch]
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            masks = create_batch_loss_masks(batch[4]).to(device)

            outputs = outputs * masks
            tgt = batch[-1] * masks
            y_p = outputs.reshape(outputs.size(0), -1)
            y_t = tgt.reshape(tgt.size(0), -1)
            loss = masked_spectral_distance(y_t, y_p).sum() / batch[0].size(0)

            val_sum += loss.item()
            if rank == 0:
                pbar.set_postfix({"val_loss": loss.item()})

    t_sum = torch.tensor(val_sum, device=device)
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    avg = t_sum.item() / (world_size * len(val_loader))
    if rank == 0:
        logging.info(f"epoch {epoch} avg val loss: {avg:.4f}")
    return avg


def save_ckpt(rank, model, optimizer, epoch, val_loss, out_dir):
    if rank != 0:
        return
    sd = model.state_dict()
    ts = datetime.now().strftime("%m%d_%H%M%S")
    fp = os.path.join(
        out_dir,
        f"model_epoch_{epoch}_val_loss_{val_loss:.4f}_{ts}.pth",
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        fp,
    )
    logging.info(f"ckpt saved: {fp}")


# ============================== 入口 ==============================
if __name__ == "__main__":
    seed_everything(seed=42, workers=True)
    resolved_port = _resolve_master_port(config.port)
    os.environ["MS2INT_MASTER_PORT"] = str(resolved_port)
    if str(config.port) != str(resolved_port):
        print(f"端口 {config.port} 被占用，自动切换到 {resolved_port}")

    torch.multiprocessing.spawn(
        train,
        args=(config.world_size, Mamba_Config),
        nprocs=config.world_size,
        join=True,
    )
