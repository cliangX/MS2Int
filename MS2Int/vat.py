"""Virtual Adversarial Training (VAT) loss module.

Based on the VAT method proposed by Google Brain:
    Miyato, T., Maeda, S., Koyama, M., & Ishii, S. (2018).
    "Virtual Adversarial Training: A Regularization Method for
    Supervised and Semi-Supervised Learning."
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(8), 1979-1993.
    https://arxiv.org/abs/1704.03976

Adapted for regression on mass spectrometry intensity prediction:
  - Perturbation target: collision energy embedding output (via forward hook).
  - Consistency metric: masked MSE on valid spectrum positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    cur: torch.nn.Module = root
    for name in path.split("."):
        cur = getattr(cur, name)
    return cur


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_flat = x.view(x.size(0), -1)
    norm = torch.norm(x_flat, p=2, dim=1, keepdim=True).clamp_min(eps)
    x_norm = x_flat / norm
    return x_norm.view_as(x)


@dataclass
class VATConfig:
    eps: float = 2.0
    xi: float = 1e-6
    ip: int = 1
    emb_name: str = "backbone.MetaEmbedding.collision_energy_embedding"


class VATLoss(nn.Module):

    def __init__(self, eps: float = 2.0, xi: float = 1e-6, ip: int = 1, emb_name: str = VATConfig.emb_name):
        super().__init__()
        self.cfg = VATConfig(eps=eps, xi=xi, ip=ip, emb_name=emb_name)

    def _mse_masked(self, pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masks_f = masks.to(dtype=pred.dtype)
        diff = (pred - target) * masks_f
        denom = masks_f.sum(dim=(1, 2)).clamp_min(1.0)
        mse_per_sample = diff.pow(2).sum(dim=(1, 2)) / denom
        return mse_per_sample.mean()

    def _forward_with_delta(
        self,
        model: torch.nn.Module,
        inst: torch.Tensor,
        charge: torch.Tensor,
        ce: torch.Tensor,
        seq: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        root = _unwrap_ddp(model)
        emb_layer = _get_submodule(root, self.cfg.emb_name)

        def hook(_module, _inputs, output):
            return output + delta

        handle = emb_layer.register_forward_hook(hook)
        try:
            return model(inst, charge, ce, seq)
        finally:
            handle.remove()

    def forward(
        self,
        model: torch.nn.Module,
        inst: torch.Tensor,
        charge: torch.Tensor,
        ce: torch.Tensor,
        seq: torch.Tensor,
        masks: torch.Tensor,
        base_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_pred is None:
            with torch.no_grad():
                base_pred = model(inst, charge, ce, seq)
        base_pred = base_pred.detach()

        # CE embedding 是 per-sample 标量，无 padding，不需要 token_mask
        root = _unwrap_ddp(model)
        emb_layer = _get_submodule(root, self.cfg.emb_name)
        with torch.no_grad():
            emb_out = emb_layer(ce)  # (B, energy_dim)

        d = torch.randn_like(emb_out)
        d = _l2_normalize(d).detach()

        # 中间迭代只为求 d 的梯度，不需要 DDP 同步，用 unwrapped model 避免冗余开销
        raw_model = _unwrap_ddp(model)
        for _ in range(max(1, int(self.cfg.ip))):
            d = d.detach()
            d.requires_grad_(True)
            delta = self.cfg.xi * d

            y_hat = self._forward_with_delta(raw_model, inst, charge, ce, seq, delta)
            div = self._mse_masked(y_hat, base_pred, masks)
            grad = torch.autograd.grad(div, d, retain_graph=False, create_graph=False)[0]

            d = _l2_normalize(grad.detach())

        r_adv = self.cfg.eps * d.detach()
        # 最终前向走 DDP 保证反向传播时梯度正确同步
        y_adv = self._forward_with_delta(model, inst, charge, ce, seq, r_adv)
        vat_loss = self._mse_masked(y_adv, base_pred, masks)
        return vat_loss

