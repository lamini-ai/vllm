# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from collections import OrderedDict
from typing import Dict, List, TypedDict
from unittest.mock import MagicMock, patch

import pytest
import safetensors
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import vllm
from vllm.config import LoRAConfig, MoMEConfig
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import get_model
from vllm.platforms import current_platform


class ContextIDInfo(TypedDict):
    lora_id: int
    context_length: str


class ContextInfo(TypedDict):
    lora: str
    context_length: str


LONG_LORA_INFOS: List[ContextIDInfo] = [{
    "lora_id": 1,
    "context_length": "16k",
}, {
    "lora_id": 2,
    "context_length": "16k",
}, {
    "lora_id": 3,
    "context_length": "32k",
}]


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(scope="session")
def mome_adapter_files():
    adapter_path = "/root/34916/checkpoints/checkpoint-60"
    if os.path.exists(adapter_path):
        return adapter_path
    else:
        raise FileNotFoundError(
            f"MoME adapter files not found at {adapter_path}. "
            "Please make sure the test files had copy from the appropriate source."
        )

@pytest.fixture
def dummy_model() -> nn.Module:
    model = nn.Sequential(
        OrderedDict([
            ("embed_tokens", nn.Embedding(128256, 4096)),

            ("layer0", nn.Sequential(OrderedDict([
                ("input_layernorm", nn.LayerNorm(4096)),
                ("self_attn", nn.Identity()),
                ("post_attention_layernorm", nn.LayerNorm(4096)),
                ("mlp", LlamaMLP(hidden_size=4096, intermediate_size=14336, hidden_act="silu")),
            ]))),

            ("layer1", nn.Sequential(OrderedDict([
                ("input_layernorm", nn.LayerNorm(4096)),
                ("self_attn", nn.Identity()),
                ("post_attention_layernorm", nn.LayerNorm(4096)),
                ("mlp", LlamaMLP(hidden_size=4096, intermediate_size=14336, hidden_act="silu")),
            ]))),

            ("norm", nn.LayerNorm(4096)),
            ("lm_head", ParallelLMHead(4096, 128256)),
            ("logits_processor", LogitsProcessor(128256)),
            ("sampler", Sampler())
        ])
    )
    model.config = MagicMock()
    return model
