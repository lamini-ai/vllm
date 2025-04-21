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
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention, ParallelLMHead
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
from vllm.mome.model_definition.lamini_index import LaminiIndex


logger = init_logger(__name__)


def _get_mome_device(base_layer: nn.Module) -> torch.device:
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # Compressed Tensor
    elif hasattr(base_layer, "weight_packed"):
        return base_layer.weight_packed.device
    # GPTQ/AWQ
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    else:
        for param in base_layer.parameters():
            return param.device
    raise ValueError(f"Unsupported get device from base layer: {base_layer}")

def get_hidden_size(layer):
    logger.debug(f"getting hidden size for layer: {layer}")
    if hasattr(layer, "attention"):
        return get_hidden_size(layer.attention)

    if hasattr(layer, "hidden_size"):
        logger.debug(f"hidden size: {layer.hidden_size} from layer.hidden_size")
        return layer.hidden_size

    def get_proj_hidden(p):
        try:
            return list(p.parameters())[0].shape[1]
        except Exception:
            return None

    for name in ["q_proj", "out_proj", "c_fc", "fc2", "gate_up_proj", "c_proj"]:
        if hasattr(layer, name):
            sub = getattr(layer, name)
            h = get_proj_hidden(sub)
            if h:
                logger.debug(f"hidden size: {h} from layer.{name}")
                return h

    if hasattr(layer, "head_size") and hasattr(layer, "num_heads"):
        hidden_size = layer.head_size * layer.num_heads
        logger.debug(f"hidden size: {hidden_size} computed from head_size * num_heads")
        return hidden_size

    raise ValueError(f"Can't determine hidden size for layer type: {type(layer)}")

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
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len
    ):
        self.indices_gpu = base_indices.to(device=self.device)
        self.sampler_indices_gpu: sampler_indices.to(device=self.device)
        self.embedding_indices_gpu = embeddings_indices.to(device=self.device)
        self.indices_len = indices_len

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
    def __init__(self, base_layer: ReplicatedLinear):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.output_size = self.base_layer.output_size
        self.device = _get_mome_device(self.base_layer)

        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor
        self.sampler_indices_gpu: torch.Tensor
        self.indices_len: List[int] = []

        self.mome_attention = None

    def create_mome_weights(
        self,
        max_loras: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.mome_config = mome_config

        lora_a_out_size = mome_config.max_mome_rank
        lora_b_out_size = self.output_size
        # self.lora_a_tensors = torch.zeros(
        #     (                
        #         max_loras,
        #         lora_a_out_size,
        #         self.input_size,
        #     ),
        #     dtype=mome_config.mome_dtype,
        #     device=self.device,
        # )
        # self.lora_b_tensors = torch.zeros(
        #     (
        #         max_loras,
        #         lora_b_out_size,
        #         lora_a_out_size,
        #     ),
        #     dtype=mome_config.mome_dtype,
        #     device=self.device,
        # )
        self.mome_attention_list = []

    def reset_mome(self, index: int):
        # self.lora_a_tensors[index] = 0
        # self.lora_b_tensors[index] = 0
        self.mome_attention_list[index] = None

    def set_mome(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        mome_index: LaminiIndex,
        mome_index_k: int,
    ):
        # Except for QKVParallelLinearWithLora and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        # assert (len(self.lora_a_tensors) == len(self.lora_b_tensors))
        self.reset_mome(index)
        # self.lora_a_tensors[index, :lora_a.shape[1], :lora_a.shape[0]].copy_(
        #                            lora_a.T, non_blocking=True)
        # self.lora_b_tensors[index, :lora_b.shape[1], :lora_b.shape[0]].copy_(
        #                            lora_b.T, non_blocking=True)
        
        self.mome_attention_list[index] = MoMEAttentionLayer(
            hidden_size=get_hidden_size(self.base_layer),
            r_value=rank,
            device=self.device,
            index=mome_index,
            index_k=mome_index_k,
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

        layer_outputs = self.base_layer(hidden_states)
        logger.debug("layer_outputs.shape:", layer_outputs.shape)
        # project the mome attention output to the same size as the transformer attention output
        mome_attention_output = self.mome_attention[0].forward(hidden_states)
        logger.debug(f"mome_attention_output shape: {mome_attention_output.shape}")
        # logger.debug(
        #     f"mome_attention_output: {mome_attention_output} {torch.histogram(mome_attention_output, bins=4)}"
        # )
        # logger.debug(
        #     f"self_attention_output: {self_attention_output} {torch.histogram(self_attention_output, bins=4)}"
        # )
 
        if isinstance(layer_outputs, tuple):
            return (layer_outputs[0] + mome_attention_output,) + layer_outputs[1:]
        else:
            return layer_outputs + mome_attention_output

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          mome_config: MoMEConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is LlamaAttention    


class MoMEAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        r_value,
        device,
        index: LaminiIndex,
        index_k,
    ):
        super().__init__()
        self.index = index
        self.index_k = index_k
        self.index_dimension = index.embedding_dimension

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
        self, hidden_states: torch.Tensor,
    ):
        # print("hidden_states.shape:", hidden_states.shape)
        query = self.project_query(hidden_states)
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
    def __init__(self, base_layer: LlamaMLP):
        super().__init__()
        self.base_layer = base_layer
        # self.hidden_size = self.base_layer.down_proj.output_size
        self.hidden_size = get_hidden_size(self.base_layer)
        self.device = _get_mome_device(self.base_layer)

        # mapping tensors
        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor
        self.sampler_indices_gpu: torch.Tensor
        self.indices_len: List[int] = []

        self.mlp_mome_in = None
        self.mlp_mome_out = None

    def _reset_parameters(self, index):
        self.mlp_mome_out[index].weight.data.zero_()

    def create_mome_weights(
        self,
        max_loras: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.mome_config = mome_config

        # lora_a_out_size = mome_config.max_mome_rank
        # lora_b_out_size = self.hidden_size
        # self.lora_a_tensors = torch.zeros(
        #     (                
        #         max_loras,
        #         lora_a_out_size,
        #         self.input_size,
        #     ),
        #     dtype=mome_config.mome_dtype,
        #     device=self.device,
        # )
        # self.lora_b_tensors = torch.zeros(
        #     (
        #         max_loras,
        #         lora_b_out_size,
        #         lora_a_out_size,
        #     ),
        #     dtype=mome_config.mome_dtype,
        #     device=self.device,
        # )

        self.mlp_mome_in = []
        self.mlp_mome_out = []

    def reset_mome(self, index: int):
        self.lora_a_tensors[index] = 0
        self.lora_b_tensors[index] = 0

    def set_mome(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        mome_index: LaminiIndex,
        mome_index_k: int,
    ):
        # assert (len(self.lora_a_tensors) == len(self.lora_b_tensors))
        self.reset_mome(index)
        # self.lora_a_tensors[index, :lora_a.shape[1], :lora_a.shape[0]].copy_(
        #                            lora_a.T, non_blocking=True)
        # self.lora_b_tensors[index, :lora_b.shape[1], :lora_b.shape[0]].copy_(
        #                            lora_b.T, non_blocking=True)
        self.mlp_mome_in[index] = nn.Linear(self.hidden_size, rank, bias=False)
        self.mlp_mome_out[index] = nn.Linear(rank, self.hidden_size, bias=False)
        self._reset_parameters(index)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(hidden_states)
        mome_in_results = self.mlp_mome_in[0](output)
        mome_out_results = self.mlp_mome_out[0](mome_in_results)
        # logger.debug(
        #     f"mome_results: {mome_results} {torch.histogram(mome_results, bins=4)}"
        # )
        # logger.debug(
        #     f"base_model_results: {base_model_results} {torch.histogram(base_model_results, bins=4)}"
        # )
        # sum the two outputs
        return output + mome_out_results

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        mome_config: MoMEConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is LlamaMLP


class LoraHeadAdaptor(BaseLayerWithMoME):
    # TODO: update LoraHeadAdaptor init to work with from_layer_logits_processor
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = self.base_layer.weight.shape
        # Add a mome attention layer

        self.device = _get_mome_device(self.base_layer)

        # mapping tensors
        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor
        self.sampler_indices_gpu: torch.Tensor
        self.indices_len: List[int] = []

        self.mlp_mome_in = []
        self.mlp_mome_out = []

    def _reset_parameters(self, index):
        self.mlp_mome_out[index].weight.data.zero_()
    
    def reset_mome(self, index: int):
        self.mlp_lora_in[index] = None
        self.mlp_lora_out[index] = None

    def set_mome(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        mome_index: LaminiIndex,
        mome_index_k: int,
    ):
        self.reset_mome(index)
        self.mlp_lora_in[index] = nn.Linear(self.hidden_size[1], rank, bias=False)
        self.mlp_lora_out[index] = nn.Linear(rank, self.hidden_size[0], bias=False)
        self._reset_parameters(index)

    # Call layer with all inputs and kwargs
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(hidden_states)
        mome_in_results = self.mlp_mome_in[0](output)
        mome_out_results = self.mlp_mome_out[0](mome_in_results)
        return output + mome_out_results

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        mome_config: MoMEConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ParallelLMHead
