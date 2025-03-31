import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.adapter_commons.layers import AdapterMapping
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_gather)
from vllm.distributed.utils import divide
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.platforms import current_platform



@dataclass
class MoMEMapping(AdapterMapping):
    is_prefill: bool = False


class AttentionLayerWithMoME(nn.Module):
    def __init__(self, base_attention: nn.Module, hidden_size: int, r: int):
        super().__init__()
        self.base_attention = base_attention
        self.r = r
        self.query_proj = nn.Linear(hidden_size, r, bias=False)
        self.value_proj = nn.Linear(r, hidden_size, bias=False)

        # runtime slot -> (K,V) embedding
        self.slots = {}  # slot_id -> dict(key=..., value=...)

    def add_mome(self, slot_id: int, key: torch.Tensor, value: torch.Tensor):
        self.slots[slot_id] = {'key': key, 'value': value}

    def forward(self, hidden_states, attention_mask=None, mome_mapping=None):
        output = self.base_attention(hidden_states, attention_mask=attention_mask)[0]

        if mome_mapping is not None:
            # slot_ids = torch.tensor(mome_mapping.index_mapping, device=hidden_states.device)
            # # [B, T, D] -> [B*T, D]
            # query = self.query_proj(hidden_states).reshape(-1, self.r)

            # slot_id = slot_ids[0].item()
            # mome_k = self.slots[slot_id]['key']
            # mome_v = self.slots[slot_id]['value']

            # attn_out = None

            # output = output + self.value_proj(attn_out).reshape_as(output)
            pass

        return (output,)