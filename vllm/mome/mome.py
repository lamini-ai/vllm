# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Dict, Union
from typing import Sequence as GenericSequence

import torch
import torch.types

from vllm.utils import is_pin_memory_available
from vllm.mome.model_definition.lamini_index import LaminiIndex


class MoMELayerWeights:
    """MoME weights for a layer composed of two low rank matrixes."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        index: LaminiIndex = None,
        index_k: int = None,
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_a = lora_a
        self.lora_b = lora_b
        
        # for MoME Attention
        self.index = index
        self.index_k = index_k

    @property
    def input_dim(self) -> int:
        return self.lora_a.shape[0]

    @property
    def output_dim(self) -> int:
        return self.lora_b.shape[1]

    @property
    def is_packed(self) -> bool:
        return False

    @classmethod
    def from_config(
        cls,
        module_name: str,
        rank: int,
    ) -> "LoRALayerWeights":
        
        return cls(module_name, rank, None, None, None, None)

    @classmethod
    def create_dummy_lora_weights(
            cls,
            module_name: str,
            input_dim: int,
            output_dim: int,
            rank: int,
            dtype: torch.dtype,
            device: torch.types.Device,
            index: LaminiIndex = None,
            index_k: int = None) -> "LoRALayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        lora_a = torch.zeros([input_dim, rank],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        lora_b = torch.zeros([rank, output_dim],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)

        return cls(
            module_name,
            rank=rank,
            lora_a=lora_a,
            lora_b=lora_b,
            index=index,
            index_k=index_k,
        )

