import os
import logging
from datetime import datetime
import socket

from hparams import get_hparams

config = get_hparams()

try:
    import setproctitle
except ModuleNotFoundError:  # pragma: no cover
    setproctitle = None

if setproctitle is not None:
    setproctitle.setproctitle(config.experiment_name)

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

from model import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
try:
    from lightning.pytorch import seed_everything
except ModuleNotFoundError:  # pragma: no cover
    def seed_everything(seed=42, workers=True):
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
from utils import *
from datasets import CustomDataset

from vat import VATLoss


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


WORLD_SIZE = config.world_size


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _resolve_master_port(preferred_port: int) -> int:
    try:
        port = int(preferred_port)
    except (TypeError, ValueError):
        port = 0

    if port > 0 and _is_port_available(port):
        return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = str(config.server)
    os.environ["MASTER_PORT"] = str(os.environ.get("MS2INT_MASTER_PORT", config.port))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, Mamba_Config):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = MambaLMHeadModel(Mamba_Config)
    load_checkpoint(config.pth, model)
    model = model.to(device)

    params_df = count_parameters(model)
    logging.info(params_df)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    vat_loss_fn = VATLoss(eps=config.vat_eps, xi=config.vat_xi, ip=config.vat_ip)

    dataset = CustomDataset(config.train_data_path)
    total_samples = len(dataset)
    train_size = int(config.train_data_size * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = CosineWarmupScheduler(optimizer, warmup=config.warmup_iters, max_iters=config.max_iters)

    best_val_loss = 99999
    max_epochs = config.max_epochs
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        if rank == 0:
            train_progress = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch + 1}/{max_epochs} Training(VAT-FT)")
        else:
            train_progress = train_loader

        for _batch_idx, batch in enumerate(train_progress):
            batch = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(batch[0], batch[1], batch[2], batch[3])

            masks = create_batch_loss_masks(batch[4]).to(device)
            t = batch[-1]
            t[t == 0] = -1

            base_pred = outputs.detach()

            outputs_sup = outputs * masks
            tgt_train_data = batch[-1] * masks
            y_outputs = outputs_sup.reshape(outputs_sup.size(0), -1)
            y_tgt_train_data = tgt_train_data.reshape(tgt_train_data.size(0), -1)
            sup_loss = masked_spectral_distance(y_tgt_train_data, y_outputs).sum()

            batch_size = batch[0].size(0)
            sup_loss = sup_loss / batch_size

            vat_loss = vat_loss_fn(model, batch[0], batch[1], batch[2], batch[3], masks, base_pred=base_pred)
            loss = sup_loss + config.vat_alpha * vat_loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_total_item = float(loss.item())
            if rank == 0:
                train_progress.set_postfix({"train_loss": loss_total_item})
            epoch_loss += loss_total_item

        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_avg = epoch_loss_tensor.item() / (world_size * len(train_loader))
        if rank == 0:
            logging.info(f"average train loss: {epoch_loss_avg}")

        val_loss = validate(rank, world_size, model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(rank, model, optimizer, epoch, val_loss, config.checkpoint_path)

    cleanup()


def validate(rank, world_size, model, val_loader, device):
    model.eval()
    val_loss = 0.0

    if rank == 0:
        val_progress = tqdm(val_loader, ncols=100, desc="Validation(VAT-FT)   ")
    else:
        val_progress = val_loader

    with torch.no_grad():
        for batch in val_progress:
            batch = [t.to(device) for t in batch]
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            masks = create_batch_loss_masks(batch[4]).to(device)

            t = batch[-1]
            t[t == 0] = -1

            outputs = outputs * masks
            tgt_train_data = batch[-1] * masks
            y_outputs = outputs.reshape(outputs.size(0), -1)
            y_tgt_train_data = tgt_train_data.reshape(tgt_train_data.size(0), -1)
            loss = masked_spectral_distance(y_tgt_train_data, y_outputs).sum()

            batch_size = batch[0].size(0)
            loss = loss / batch_size

            val_loss += loss.item()
            if rank == 0:
                val_progress.set_postfix({"val_loss": loss.item()})

    epoch_val_loss_tensor = torch.tensor(val_loss, device=device)
    dist.all_reduce(epoch_val_loss_tensor, op=dist.ReduceOp.SUM)
    epoch_val_loss = epoch_val_loss_tensor.item() / (world_size * len(val_loader))

    if rank == 0:
        logging.info(f"average val loss: {epoch_val_loss}")

    return epoch_val_loss


def save_checkpoint(rank, model, optimizer, epoch, val_loss, checkpoint_path):
    if rank == 0:
        model_state_dict = model.state_dict()
        current_time = datetime.now().strftime("%m%d_%H%M%S")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(checkpoint_path, f"model_epoch_{epoch}_val_loss_{val_loss:.4f}_{current_time}.pth"),
        )
        logging.info(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss:.4f}")


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)
    resolved_port = _resolve_master_port(config.port)
    os.environ["MS2INT_MASTER_PORT"] = str(resolved_port)
    if int(config.port) != resolved_port:
        print(f"端口 {config.port} 被占用，已自动切换到 {resolved_port}")

    torch.multiprocessing.spawn(train, args=(WORLD_SIZE, Mamba_Config), nprocs=WORLD_SIZE, join=True)
