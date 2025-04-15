import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import numpy as np
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

import inspect
from vllm.mome.utils import get_hidden_size
from vllm.mome.model_definition.lamini_index import LaminiIndex


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
        adapter_model: Optional[torch.Tensor]
    ):
        """Overwrites mome tensors at index."""
        ...

    def set_mapping(
        self,
        mapping,
    ):
        self.mapping = mapping
        # self.punica_wrapper: PunicaWrapperBase = punica_wrapper

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


class BaseMoMEAttentionLayer(BaseLayerWithMoME):
    def __init__(
        self,
        base_layer,
        embeddings,
        index,
        r_value,
        mome_embedding_seq_length,
        index_k,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.device = _get_mome_device(self.base_layer)

        # Add a mome attention layer
        #print("layer:", dir(layer))
        self.mome_attention = MoMEAttentionLayer(
            hidden_size=get_hidden_size(base_layer),
            r_value=r_value,
            mome_embedding_seq_length=mome_embedding_seq_length,
            device=self.device,
            embeddings=embeddings,
            index=index,
            index_k=index_k,
        )

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        # logger.debug(f"hidden states dtype: {hidden_states.dtype}")
        # if attention_mask is not None:
        #    logger.debug(f"attention mask dtype: {attention_mask.dtype}")

        # if position_ids is not None:
        #    logger.debug(f"position ids dtype: {position_ids.dtype}")

        # if past_key_value is not None:
        #    logger.debug(f"past_key_value dtype: {past_key_value}")

        layer_outputs = self.base_layer.apply(hidden_states)
        print("layer_outputs.shape:", layer_outputs.shape)
        # project the mome attention output to the same size as the transformer attention output
        mome_attention_output = self.mome_attention(hidden_states)

        # logger.debug(
        #     f"mome_attention_output: {mome_attention_output} {torch.histogram(mome_attention_output, bins=4)}"
        # )
        # logger.debug(
        #     f"self_attention_output: {self_attention_output} {torch.histogram(self_attention_output, bins=4)}"
        # )

        # print("type layer_outputs: ", type(layer_outputs))
        if isinstance(layer_outputs, tuple):
            return (layer_outputs[0] + mome_attention_output,) + layer_outputs[1:]
        else:
            return layer_outputs + mome_attention_output


class MoMEAttentionLayer(nn.Module):
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
        # device = "cuda"
        self.query_projection_lora_in = nn.Linear(hidden_size, r_value, bias=False, device=device)
        self.query_projection_lora_out = nn.Linear(
            r_value, self.index_dimension, bias=False, device=device
        )

        # A linear layer to project the value into the space of the original hidden size
        self.value_projection_lora_in = nn.Linear(
            self.index_dimension, r_value, bias=False, device=device
        )
        self.value_projection_lora_out = nn.Linear(r_value, hidden_size, bias=False, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        self.value_projection_lora_out.weight.data.zero_()
        self.query_projection_lora_out.weight.data.zero_()

    # Call layer with all inputs and kwargs
    def forward(
        self, input_: torch.Tensor
    ):
        # print("hidden_states.shape:", hidden_states.shape)
        query = self.project_query(input_)
        # print("query.shape:", query.shape)
        key, value = self.get_key_and_value(query)

        # logger.debug(f"query shape: {query.shape}, type: {query.dtype}")
        # logger.debug(f"key shape: {key.shape}, type: {key.dtype}")
        # logger.debug(f"value shape: {value.shape}, type: {value.dtype}")

        # convert key to the dtype of the query
        target_dtype = input_.dtype
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

    def project_value(self, value):
        value = self.value_projection_lora_in(value)
        value = self.value_projection_lora_out(value)
        return value

    def project_query(self, hidden_states):
        original_dtype = hidden_states.dtype
        # assign into the mome embedding space
        query = self.query_projection_lora_in(hidden_states)
        query = self.query_projection_lora_out(query)
        query = query.to(original_dtype)
        return query

    def get_key_and_value(self, query):
        key, value, indices = self.get_key_and_value_from_index(query)
        self.embeddings["embedding_indices"][self.embedding_index] = indices

        # Get the sequence length
        batch_size = key.shape[0]
        k_times_sequence_length = key.shape[1]

        # If we are in training mode, then we need to copy the key and value
        # to the parameter buffer so that back propogation works
        if self.training:
            with torch.no_grad():
                self.key_embedding[:batch_size, :k_times_sequence_length, :].copy_(key)
                self.value_embedding[:batch_size, :k_times_sequence_length, :].copy_(
                    value
                )
                del key
                del value
            return (
                self.key_embedding[:batch_size, :k_times_sequence_length, :],
                self.value_embedding[:batch_size, :k_times_sequence_length, :],
            )
        else:
            return key, value

    def get_key_and_value_from_index(self, query):
        print("query size: ", query.shape)
        # print("query.shape[0]: ", query.shape[0])
        # print("query.shape[1]: ", query.shape[1])
        if query.dim() == 2:
            batch_size = 1
            sequence_length = query.shape[0]
            embedding_dimension = query.shape[1]
        else:
            batch_size = query.shape[0]
            sequence_length = query.shape[1]
            embedding_dimension = query.shape[2]

        # logger.debug(f"batch_size: {batch_size}")
        # logger.debug(f"sequence_length: {sequence_length}")
        # logger.debug(f"embedding_dimension: {embedding_dimension}")

        device = query.device
        original_dtype = query.dtype

        query_new = query.view(batch_size * sequence_length, embedding_dimension)

        # get the key and value from the index, no gradients
        with torch.no_grad():
            # convert query to float32
            query_new = query_new.float()

            # convert query to a numpy array
            query_new = query_new.cpu().numpy()

            key, value, indices = self.index.get_key_and_value(
                query_new, k=self.index_k
            )
            # convert key and values, which are lists, to numpy arrays
            key = np.array(key)
            value = np.array(value)

            # convert key and value to torch tensors
            key = torch.from_numpy(key).to(device, dtype=original_dtype)
            value = torch.from_numpy(value).to(device, dtype=original_dtype)

            # logger.debug(f"key shape: {key.shape}")

            key = key.view(
                batch_size, self.index_k * sequence_length, embedding_dimension
            )
            value = value.view(
                batch_size, self.index_k * sequence_length, embedding_dimension
            )

        return key, value, indices


class LoraMLPAdaptor(BaseLayerWithMoME):
    def __init__(self, base_layer: LinearBase, r_value: int):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_mome_device(self.base_layer)
        # self.output_size = self.base_layer.output_size
        # Get the hidden size
        hidden_size = get_hidden_size(base_layer)
        # hidden_size = self.input_size
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
    # TODO: update LoraHeadAdaptor init to work with from_layer_logits_processor
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
