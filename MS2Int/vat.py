from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """兼容 DDP：返回真实的模型对象。"""
    return model.module if hasattr(model, "module") else model


def _get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """按 'a.b.c' 的路径获取子模块（找不到会抛异常，便于定位配置问题）。"""
    cur: torch.nn.Module = root
    for name in path.split("."):
        cur = getattr(cur, name)
    return cur


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对每个样本做 L2 归一化（在除 batch 维之外的所有维度上）。"""
    x_flat = x.view(x.size(0), -1)
    norm = torch.norm(x_flat, p=2, dim=1, keepdim=True).clamp_min(eps)
    x_norm = x_flat / norm
    return x_norm.view_as(x)


@dataclass
class VATConfig:
    eps: float = 2.0
    xi: float = 1e-6
    ip: int = 1
    emb_name: str = "backbone.AminoAcidEmbedding.aa_embedding"


class VATLoss(nn.Module):
    """
    VAT（虚拟对抗训练）损失：回归输出版本（MSE 一致性约束）。

    关键设计：
    - 扰动注入点：序列 embedding 的输出（nn.Embedding forward hook），不改模型 forward 签名。
    - 距离度量：在训练掩码 masks 上做 MSE，一致性目标为 base_pred（detach）。
    """

    def __init__(self, eps: float = 2.0, xi: float = 1e-6, ip: int = 1, emb_name: str = VATConfig.emb_name):
        super().__init__()
        self.cfg = VATConfig(eps=eps, xi=xi, ip=ip, emb_name=emb_name)

    def _mse_masked(self, pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        在 masks 指定的有效位置上计算按样本归一化的 MSE，并对 batch 求平均。

        pred/target: (B, 29, 31)
        masks: (B, 29, 31) 取值为 0/1
        """
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
        """在 embedding 输出上注入 delta 后做一次前向。"""
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
        """
        计算 VAT 损失（返回标量）。

        参数：
        - model: 可为普通模型或 DDP 包装模型
        - inst/charge/ce/seq: 模型输入（与训练脚本一致）
        - masks: (B,29,31) 损失掩码
        - base_pred: 基准输出（建议传入 outputs.detach() 以避免重复前向）
        """
        if base_pred is None:
            with torch.no_grad():
                base_pred = model(inst, charge, ce, seq)
        base_pred = base_pred.detach()

        # 仅扰动真实 token（padding token 为 0）
        token_mask = (seq != 0).to(dtype=base_pred.dtype).unsqueeze(-1)  # (B,30,1)

        root = _unwrap_ddp(model)
        emb_layer = _get_submodule(root, self.cfg.emb_name)
        with torch.no_grad():
            emb_out = emb_layer(seq)  # (B,30,128)

        d = torch.randn_like(emb_out)
        d = d * token_mask
        d = _l2_normalize(d).detach()

        # power iteration：估计最坏方向
        for _ in range(max(1, int(self.cfg.ip))):
            d = d.detach()
            d.requires_grad_(True)
            delta = self.cfg.xi * d
            delta = delta * token_mask

            y_hat = self._forward_with_delta(model, inst, charge, ce, seq, delta)
            div = self._mse_masked(y_hat, base_pred, masks)
            grad = torch.autograd.grad(div, d, retain_graph=False, create_graph=False)[0]

            d = _l2_normalize(grad.detach()) * token_mask

        r_adv = self.cfg.eps * d.detach()
        y_adv = self._forward_with_delta(model, inst, charge, ce, seq, r_adv)
        vat_loss = self._mse_masked(y_adv, base_pred, masks)
        return vat_loss

