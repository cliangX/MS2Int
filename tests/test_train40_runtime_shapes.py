from __future__ import annotations

import importlib
import sys
import types

import torch
import pytest

from .conftest import write_h5_rows


def test_data_read_outputs_length_40_contract(tmp_path):
    h5_path = tmp_path / "sample_train40.h5"
    row = {
        "annotate": "A" * 40,
        "Length": 40,
        "Charge": 6,
        "collision_energy": 32.0,
        "Fragmentation": "HCD",
    }
    write_h5_rows(h5_path, [row], include_train_data=True)

    from MS2Int.preprocess import data_read

    instrument, charge, ce, sequence, length, train_data = data_read(str(h5_path), 0)

    assert int(length.item()) == 40
    assert tuple(sequence.shape) == (40,)
    assert int(charge.item()) == 5
    assert int(ce.item()) == 5
    assert tuple(train_data.shape) == (39, 41)


def test_create_batch_loss_masks_matches_39x41_contract():
    from MS2Int.utils import create_batch_loss_masks

    masks = create_batch_loss_masks([40, 5])

    assert tuple(masks.shape) == (2, 39, 41)
    assert int(masks[0].sum().item()) > int(masks[1].sum().item())


def test_meta_embedding_model_expands_to_length_40():
    from MS2Int.utils import MetaEmbeddingModel

    model = MetaEmbeddingModel()
    output = model(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))

    assert tuple(output.shape) == (1, 40, 128)


def test_model_shape_constants_follow_39x41_contract(monkeypatch):
    fake_utils = types.ModuleType("MS2Int.utils")

    class FakeMetaEmbeddingModel(torch.nn.Module):
        def forward(self, instrument_idx, charge_idx, collision_energy_idx):
            return torch.ones((instrument_idx.shape[0], 40, 128))

    class FakeAminoAcidEmbedding(torch.nn.Module):
        def __init__(self, amino_acid_dim=128):
            super().__init__()
            self.embedding = torch.nn.Embedding(64, amino_acid_dim)

        def forward(self, aa_idx):
            return self.embedding(aa_idx)

    fake_utils.MetaEmbeddingModel = FakeMetaEmbeddingModel
    fake_utils.AminoAcidEmbedding = FakeAminoAcidEmbedding

    fake_config_mod = types.ModuleType("mamba_ssm.models.config_mamba")

    class FakeMambaConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_config_mod.MambaConfig = FakeMambaConfig

    class FakeMixer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, residual=None, inference_params=None):
            return hidden_states, hidden_states if residual is None else residual

    class FakeBlock(torch.nn.Module):
        def __init__(self, d_model, mixer_cls, mlp_cls, norm_cls=None, **kwargs):
            super().__init__()
            self.mixer = mixer_cls()

        def forward(self, hidden_states, residual=None, inference_params=None):
            return self.mixer(
                hidden_states, residual=residual, inference_params=inference_params
            )

    fake_block_mod = types.ModuleType("mamba_ssm.modules.block")
    fake_block_mod.Block = FakeBlock

    for module_name in [
        "mamba_ssm.modules.mamba_simple",
        "mamba_ssm.modules.mamba2",
        "mamba_ssm.modules.mha",
        "mamba_ssm.modules.mlp",
    ]:
        mod = types.ModuleType(module_name)
        if module_name.endswith("mamba_simple"):
            mod.Mamba = FakeMixer
        elif module_name.endswith("mamba2"):
            mod.Mamba2 = FakeMixer
        elif module_name.endswith("mha"):
            mod.MHA = FakeMixer
        else:
            mod.GatedMLP = FakeMixer
        monkeypatch.setitem(sys.modules, module_name, mod)

    fake_generation_mod = types.ModuleType("mamba_ssm.utils.generation")
    fake_generation_mod.GenerationMixin = object

    monkeypatch.setitem(sys.modules, "MS2Int.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "mamba_ssm.models.config_mamba", fake_config_mod)
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules.block", fake_block_mod)
    monkeypatch.setitem(sys.modules, "mamba_ssm.utils.generation", fake_generation_mod)

    import MS2Int.model as model_module

    importlib.reload(model_module)

    config = fake_config_mod.MambaConfig(
        d_model=32,
        n_layer=1,
        d_intermediate=0,
        ssm_cfg={},
        attn_layer_idx=[],
        attn_cfg={},
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
    )

    model = model_module.MambaLMHeadModel(config)

    assert tuple(model.backbone.transformation_matrix.shape) == (39, 40)
    assert model.lm_head.out_features == 41


def test_load_checkpoint_rejects_old_output_contract_with_clear_message(tmp_path):
    from MS2Int.utils import load_checkpoint

    checkpoint_path = tmp_path / "old_contract.pth"
    torch.save(
        {
            "model_state_dict": {"weight": torch.randn(31, 4)},
            "epoch": 3,
            "val_loss": 0.12,
        },
        checkpoint_path,
    )

    model = torch.nn.Linear(4, 41, bias=False)

    with pytest.raises(RuntimeError, match="40aa/39x41"):
        load_checkpoint(str(checkpoint_path), model)
