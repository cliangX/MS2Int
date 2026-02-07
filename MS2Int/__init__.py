"""
VAT（Virtual Adversarial Training，虚拟对抗训练）模块（仅用于训练阶段）。

当前实现为回归输出版本：使用 MSE 作为一致性约束，
并且仅对序列 embedding（AminoAcidEmbedding.aa_embedding）的输出注入扰动。
"""

from .vat import VATLoss

__all__ = ["VATLoss"]

