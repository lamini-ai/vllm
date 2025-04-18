# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from vllm.config import MoMEConfig
from vllm.mome.layers import (LoraMLPAdaptor, BaseMoMEAttentionLayer)
from vllm.mome.mome import MoMELayerWeights
from vllm.mome.models import (MoMEMapping, MoMEModel, MoMEModelManager)
from vllm.mome.request import MoMERequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
                                      WorkerLoRAManager)
from vllm.model_executor.models.llama import LlamaMLP
from vllm.platforms import current_platform

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}

EMBEDDING_PADDING_MODULES = ["lm_head"]

DEVICES = ([
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
] if current_platform.is_cuda_alike() else ["cpu"])

'''
def create_mome(mome_id: int, model: nn.Module, sub_modules: List[str],
                device: torch.device) -> MoMEModel:
    momes: Dict[str, MoMELayerWeights] = {}
    for name in sub_modules:
        w = model.get_submodule(name).weight
        momes[name] = MoMELayerWeights(
            name,
            8,
            16,
            torch.rand([w.shape[1], 8], device=device),
            torch.rand([8, w.shape[0]], device=device),
        )
    return MoMEModel(mome_id, 8, momes)
'''

def test_replace_submodules(dist_init, dummy_model):
    model = dummy_model
    model.supported_mome_modules = ["layer0.mlp",]
    model.packed_modules_mapping = {}
    manager = MoMEModelManager(
        model, 1, 1,
        MoMEConfig(max_mome_rank=8, max_momes=8, max_cpu_momes=8),
        torch.device(DEVICES[0]))
    model = manager.model


    assert isinstance(model.get_submodule("layer0.mlp"),
                      LoraMLPAdaptor)
    assert isinstance(model.get_submodule("layer1.mlp"),
                      LlamaMLP)

'''
@pytest.mark.parametrize("device", DEVICES)
def test_mome_model_manager(dist_init, dummy_model, device):
    model = dummy_model
    model.supported_lora_modules = ["dense1", "dense2", "lm_head"]
    model.packed_modules_mapping = {}
    model_lora1 = create_lora(1,
                              model, ["layer1.dense1", "dense2", "lm_head"],
                              device=device)
    model_lora2 = create_lora(2,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    model_lora3 = create_lora(3,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    manager = LoRAModelManager(model,
                               2,
                               2,
                               2,
                               MoMEConfig(max_lora_rank=8,
                                          max_cpu_loras=3,
                                          max_loras=2),
                               device=device)
    assert all(x is None for x in manager.lora_index_to_id)
    assert manager.add_adapter(model_lora1)
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_adapter(model_lora1)
    assert not manager.activate_adapter(1)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_adapter(model_lora2)
    assert not manager.activate_adapter(2)
    assert manager.add_adapter(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    with pytest.raises(ValueError):
        assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_adapter(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_adapter(model_lora2.id)
    assert manager.remove_adapter(model_lora1.id)
    assert not manager.remove_adapter(model_lora1.id)
    assert manager.add_adapter(model_lora1)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] is None
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] is None
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2

    assert manager.device == device
    assert manager.punica_wrapper.device == device
'''