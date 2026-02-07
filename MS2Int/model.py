import math
from functools import partial
import json
import os
import copy

from collections import namedtuple
from utils import MetaEmbeddingModel, AminoAcidEmbedding


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba2")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        # 兼容环境缺少 causal-conv1d 的情况：
        # - mamba_ssm 的 Mamba2 默认可能走 mem-efficient/combined 的 Triton 路径，
        #   该路径依赖 causal-conv1d；缺失时会在 forward 时报 NoneType is not callable。
        # - 这里在未显式指定时，根据依赖是否存在自动关闭该路径，保证推理可运行。
        if ssm_layer == "Mamba2":
            ssm_cfg.setdefault("use_mem_eff_path", causal_conv1d_fn is not None)
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block




def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class BiDirectionMixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        #vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        '''
        这里要改成自己的embedding
        '''
        #self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.MetaEmbedding = MetaEmbeddingModel()
        self.AminoAcidEmbedding = AminoAcidEmbedding()
        self.transformation_matrix = nn.Parameter(torch.randn(29, 30))
        self.gate = nn.Linear(2*self.d_model, 1,)
        self.expansion_layer = nn.Linear(128, d_model)  # 新增线性层
        
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.forward_layers = nn.ModuleList(
            [
                create_block(
                    self.d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        self.backward_layers = nn.ModuleList(
            [
                create_block(
                    self.d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.hidden_fc = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for i in range(n_layer)]
        ) 

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            self.d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, instrument_tensor, charge_tensor, collision_energy_tensor,sequence_tensor,
                 inference_params=None, **mixer_kwargs):
        
        '''
        自定义embedding
        '''
        
        metaembedding = self.MetaEmbedding(instrument_tensor, charge_tensor, collision_energy_tensor)
        aaembedding = self.AminoAcidEmbedding(sequence_tensor)
        midlle = metaembedding * aaembedding #(B, 30, 128)
        # # 使用新增的线性层进行维度扩展
        B, T, _ = midlle.shape
        # last = self.expansion_layer(midlle.view(-1, 128)).view(B, T, d_model)  # (B, 30, d_model)
        # hidden_states = torch.matmul(self.transformation_matrix.unsqueeze(0).expand(midlle.size(0), -1, -1),
        #                                 last)  # (B, 29, d_model)

        # embedding = torch.zeros_like(hidden_states) 
        # gate = self.gate(torch.cat([hidden_states, embedding], dim=-1)).sigmoid()
        # hidden_states = hidden_states * gate + embedding * (1 - gate)
        # 假设你已经检查了 midlle 的内存布局是连续的
        midlle_flat = midlle.view(-1, 128)  # 确保数据是连续的
        last = self.expansion_layer(midlle_flat).view(B, T, self.d_model)  # (B, 30, d_model)
        hidden_states = torch.bmm(self.transformation_matrix.unsqueeze(0).expand(B, -1, -1), last)
        embedding = torch.zeros_like(hidden_states)  # 如果有可能，尝试避免这种初始化
        hidden_and_embedding = torch.cat([hidden_states, embedding], dim=-1)
        gate = self.gate(hidden_and_embedding).sigmoid()
        hidden_states = hidden_states * gate + embedding * (1 - gate)
        


        residual = None
        for f_layer, b_layer, h_fc in zip(
            self.forward_layers, self.backward_layers, self.hidden_fc
        ):
            hidden_states_f, residual_f = f_layer(
                hidden_states, residual, inference_params=inference_params
            )
            flip_residual = residual.flip([1]) if residual is not None else None
            hidden_states_b, residual_b = b_layer(
                hidden_states.flip([1]), flip_residual, inference_params=inference_params
            )
            hidden_states = h_fc(torch.cat([hidden_states_f, hidden_states_b.flip([1])], dim=-1))
            residual = 0.5 * (residual_f + residual_b.flip([1]))




        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states



    

class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate

        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}
               
        super().__init__()
 
        self.backbone = BiDirectionMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        output_dim = 31
        ## 需要改成自己的output线性输出
        self.lm_head = nn.Linear(d_model, output_dim , bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        #self.tie_weights()

    # def tie_weights(self):
    #     if self.config.tie_embeddings:
    #         self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, instrument_tensor, charge_tensor, collision_energy_tensor, sequence_tensor, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(instrument_tensor, charge_tensor, collision_energy_tensor, sequence_tensor, 
                                      inference_params=inference_params, **mixer_kwargs)
        

        lm_logits = self.lm_head(hidden_states)
        

        return lm_logits
