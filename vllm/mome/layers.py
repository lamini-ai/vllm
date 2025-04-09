import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.adapter_commons.layers import AdapterMapping
from vllm.config import MoMEConfig
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
from vllm.logger import init_logger


logger = init_logger(__name__)


def _get_mome_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-mome-on-quantized-models/vllm/mome/layers.py#L34
    """Returns the device for where to place the MoME tensors."""
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # Compressed Tensor
    elif hasattr(base_layer, "weight_packed"):
        return base_layer.weight_packed.device
    # GPTQ/AWQ
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # marlin
    elif hasattr(base_layer, "B"):
        return base_layer.B.device
    # HQQ marlin
    elif hasattr(base_layer, "W_q"):
        return base_layer.W_q.device
    else:
        raise ValueError(f"Unsupported base layer: {base_layer}")

@dataclass
class MoMEMapping(AdapterMapping):
    is_prefill: bool = False

class BaseLayerWithMoME(nn.Module):

    def create_mome_weights(
        self,
        max_momes: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """Initializes mome matrices."""
        ...

    def reset_mome(self, index: int):
        """Resets the mome weights at index back to 0."""
        ...

    def set_mome(
        self,
        index: int,
        mome_a: torch.Tensor,
        mome_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        """Overwrites mome tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        mome_config: MoMEConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this MoME layer."""
        raise NotImplementedError



class MoMEAttentionLayer(BaseLayerWithMoME):

    def __init__(
        self,
        hidden_size,
        r_value,
        mome_embedding_seq_length,
        device,
        embeddings,
        index: LaminiIndex,
        index_k,
    ):
        super().__init__()
        self.index = index
        self.index_k = index_k
        self.index_dimension = index.embedding_dimension

        self.embeddings = embeddings

        self.key_embedding = nn.Parameter(
            torch.zeros(
                1,
                mome_embedding_seq_length * self.index_k,
                self.index_dimension,
            )
        )

        self.value_embedding = nn.Parameter(
            torch.zeros(
                1,
                mome_embedding_seq_length * self.index_k,
                self.index_dimension,
            )
        )

        self.embedding_index = len(embeddings["key_embeddings"])

        self.embeddings["key_embeddings"].append(self.key_embedding)
        self.embeddings["value_embeddings"].append(self.value_embedding)
        self.embeddings["embedding_indices"].append([])

        # A linear layer to project the query into the space of the index
        device = "cuda"
        self.query_projection_mome_in = nn.Linear(hidden_size, r_value, bias=False, device=device)
        self.query_projection_mome_out = nn.Linear(
            r_value, self.index_dimension, bias=False, device=device
        )

        # A linear layer to project the value into the space of the original hidden size
        self.value_projection_mome_in = nn.Linear(
            self.index_dimension, r_value, bias=False, device=device
        )
        self.value_projection_mome_out = nn.Linear(r_value, hidden_size, bias=False, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        self.value_projection_mome_out.weight.data.zero_()
        self.query_projection_mome_out.weight.data.zero_()

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # print("hidden_states.shape:", hidden_states.shape)
        query = self.get_query(hidden_states)
        # print("query.shape:", query.shape)
        key, value = self.get_key_and_value(query)

        # logger.debug(f"query shape: {query.shape}, type: {query.dtype}")
        # logger.debug(f"key shape: {key.shape}, type: {key.dtype}")
        # logger.debug(f"value shape: {value.shape}, type: {value.dtype}")

        # convert key to the dtype of the query
        target_dtype = hidden_states.dtype
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

        # project the mome attention output to the same size as the transformer attention output
        mome_attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            # attn_mask=attention_mask,
            dropout_p=0.1,
            is_causal=True,
            scale=None,
        )

        projected_mome_attention_output = self.project_value(mome_attention_output)

        return projected_mome_attention_output

class LoraMLPAdaptor(BaseLayerWithMoME):
    def __init__(self, base_layer: LinearBase, r_value: int):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_mome_device(self.base_layer)
        # self.output_size = self.base_layer.output_size
        # Get the hidden size
        # hidden_size = get_hidden_size(layer)
        hidden_size = self.input_size
        # Add a mome attention layer
        self.mlp_mome_in = nn.Linear(hidden_size, r_value, bias=False)
        self.mlp_mome_out = nn.Linear(r_value, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_mome_out.weight.data.zero_()


    # Call layer with all inputs and kwargs
    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward of ReplicatedLinearWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output = self.apply(input_, bias)

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias
    
    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        mome_in_results = self.mlp_mome_in(output)
        mome_results = self.mlp_mome_out(mome_in_results)
        # logger.debug(
        #     f"mome_results: {mome_results} {torch.histogram(mome_results, bins=4)}"
        # )
        # logger.debug(
        #     f"base_model_results: {base_model_results} {torch.histogram(base_model_results, bins=4)}"
        # )
        # sum the two outputs
        layer_and_adaptor_sum = output + mome_results
        return layer_and_adaptor_sum

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        mome_config: MoMEConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ReplicatedLinear


class LoraHeadAdaptor(BaseLayerWithMoME):
    def __init__(self, base_layer: LinearBase, r_value: int):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_mome_device(self.base_layer)
        # self.output_size = self.base_layer.output_size
        # Get the hidden size
        hidden_size = self.base_layer.weight.shape
        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(hidden_size[1], r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, hidden_size[0], bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_mome_out.weight.data.zero_()

    # Call layer with all inputs and kwargs
    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward of ReplicatedLinearWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output = self.apply(input_, bias)

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias
    
    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        lora_in_results = self.mlp_lora_in(output)
        lora_results = self.mlp_lora_out(lora_in_results)
        return output + lora_results

    # @property
    # def weight(self):
    #     return self.layer.weight
    # @property
    # def bias(self):
    #     return self.layer.bias

