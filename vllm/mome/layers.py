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
from vllm.mome.mome import MoMELayerWeights
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


class MoMEAttentionLayer(BaseLayerWithMoME):
    def __init__(self, base_layer: LlamaAttention):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = self.base_layer.hidden_size
        self.device = _get_mome_device(self.base_layer)

        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor
        self.sampler_indices_gpu: torch.Tensor
        self.indices_len: List[int] = []

        # self.index = None
        # self.index_k = 1
        # self.index_dimension = None

    def create_mome_weights(
        self,
        max_loras: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.mome_config = mome_config

        lora_a_out_size = mome_config.max_mome_rank
        # index_dimension = mome_config.embedding_dimension
        # TODO: get the index dimension from the mome_config
        index_dimension = 384
        lora_b_out_size = self.hidden_size

        self.query_proj_lora_a_tensors = torch.zeros(
            (                
                max_loras,
                lora_a_out_size,
                lora_b_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )
        self.query_proj_lora_b_tensors = torch.zeros(
            (
                max_loras,
                lora_b_out_size,
                index_dimension,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )
        self.value_proj_lora_a_tensors = torch.zeros(
            (
                max_loras,
                index_dimension,
                lora_a_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )
        self.value_proj_lora_b_tensors = torch.zeros(
            (
                max_loras,
                lora_a_out_size,
                lora_b_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )

    def reset_mome(self, index: int):
        self.query_proj_lora_a_tensors[index] = 0
        self.query_proj_lora_b_tensors[index] = 0
        self.value_proj_lora_a_tensors[index] = 0
        self.value_proj_lora_b_tensors[index] = 0

    def set_mome(
        self,
        index: int,
        module_mome: MoMELayerWeights
    ):
        # Except for QKVParallelLinearWithLora and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.query_proj_lora_a_tensors) == len(self.query_proj_lora_b_tensors))
        assert (len(self.value_proj_lora_a_tensors) == len(self.value_proj_lora_b_tensors))
        self.reset_mome(index)
        
        self.rank = module_mome.rank
        self.index = module_mome.index
        self.index_k = module_mome.index_k
        self.index_dimension = self.index.embedding_dimension
        query_proj_lora_a = module_mome.query_proj_lora_a
        query_proj_lora_b = module_mome.query_proj_lora_b
        value_proj_lora_a = module_mome.value_proj_lora_a
        value_proj_lora_b = module_mome.value_proj_lora_b

        self.query_proj_lora_a_tensors[index, 
                                        :query_proj_lora_a.shape[1], 
                                        :query_proj_lora_a.shape[0]].copy_(query_proj_lora_a.T, non_blocking=True)
        self.query_proj_lora_b_tensors[index, 
                                        :query_proj_lora_b.shape[1], 
                                        :query_proj_lora_b.shape[0]].copy_(query_proj_lora_b.T, non_blocking=True)
        self.value_proj_lora_a_tensors[index,
                                        :value_proj_lora_a.shape[1], 
                                        :value_proj_lora_a.shape[0]].copy_(value_proj_lora_a.T, non_blocking=True)
        self.value_proj_lora_b_tensors[index,
                                        :value_proj_lora_b.shape[1], 
                                        :value_proj_lora_b.shape[0]].copy_(value_proj_lora_b.T, non_blocking=True)

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        logger.info("hidden_states.shape: %s, hidden_states.dtype: %s", hidden_states.shape, hidden_states.dtype)

        layer_outputs = self.base_layer(hidden_states=hidden_states, **kwargs)
        logger.info("base_layer outputs.shape: %s", layer_outputs.shape)

        # project the mome attention output to the same size as the transformer attention output
        mome_attention_output = self.mome_forward(hidden_states)
        logger.info(f"mome_attention_output shape %s:", mome_attention_output.shape)
        # logger.debug(
        #     f"mome_attention_output: {mome_attention_output} {torch.histogram(mome_attention_output, bins=4)}"
        # )
        # logger.debug(
        #     f"self_attention_output: {self_attention_output} {torch.histogram(self_attention_output, bins=4)}"
        # )
        if layer_outputs.dtype != mome_attention_output.dtype:
            assert mome_attention_output.shape[0] == 1
            mome_attention_output = mome_attention_output.squeeze(0)
    
        if isinstance(layer_outputs, tuple):
            output = (layer_outputs[0] + mome_attention_output,) + layer_outputs[1:]
        else:
            output = layer_outputs + mome_attention_output
        logger.info(f"output shape: {output.shape}")
        return output

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          mome_config: MoMEConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is LlamaAttention    

    # Call layer with all inputs and kwargs
    def mome_forward(
        self, hidden_states: torch.Tensor,
        **kwargs,
    )-> torch.Tensor:
        query = self.project_query(hidden_states)
        logger.info("query.shape: %s, query.dtype: %s", query.shape, query.dtype)

        key, value = self.get_key_and_value(query)
        logger.info("key.shape: %s, key.dtype: %s", key.shape, key.dtype)
        logger.info("value.shape: %s, value.dtype: %s", value.shape, value.dtype)

        # convert key to the dtype of the query
        logger.info("start to convert key and value to the original dtype")
        target_dtype = hidden_states.dtype
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)
        logger.info("query,key,value to original dtype: %s success", target_dtype)

        # project the mome attention output to the same size as the transformer attention output
        try:
            mome_attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=0.1,
                is_causal=True,
                scale=None,
            )
        except RuntimeError as e:
            logger.error("scaled_dot_product_attention failed. " \
                            "query: %s, key: %s, value: %s", query.shape, key.shape, value.shape)
            raise e

        output = self.project_value(mome_attention_output)
        logger.info("project_value success. output dtype: %s ", output.dtype)
        return output

    def scaled_dot_product_attention_custom(self, query, key, value, attn_mask=None, dropout_p=0.0):
        """
        scaled dot product attention。
        
        query: [B, S_q, D]
        key:   [B, S_k, D]
        value: [B, S_k, D]
        attn_mask: [B, S_q, S_k] (bool, True 表示 mask 掉)
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)  # [1, S_q, D]
        if key.dim() == 2:
            key = key.unsqueeze(0)
        if value.dim() == 2:
            value = value.unsqueeze(0)

        B, S_q, D = query.shape
        S_k = key.shape[1]

        # Step 1: Attention scores = Q x K^T / sqrt(D)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (D ** 0.5)  # [B, S_q, S_k]

        # Step 2: Apply mask (if any)
        if attn_mask is not None:
            # mask: True = mask掉，需要变成 -inf，才能 softmax 成 0
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # Step 3: Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, S_q, S_k]

        # Step 4: Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Step 5: Attention output = weights x V
        output = torch.matmul(attn_weights, value)  # [B, S_q, D]

        return output

    def project_value(self, value):
        original_dtype = value.dtype
        value = F.linear(value, self.value_proj_lora_a_tensors[0], bias=None)
        value = F.linear(value, self.value_proj_lora_b_tensors[0], bias=None)
        value = value.to(original_dtype)
        return value

    def project_query(self, hidden_states):
        original_dtype = hidden_states.dtype
        query = F.linear(hidden_states, self.query_proj_lora_a_tensors[0], bias=None)
        query = F.linear(query, self.query_proj_lora_b_tensors[0], bias=None)
        query = query.to(original_dtype)
        return query

    def get_key_and_value(self, query):
        key, value, indices = self.get_key_and_value_from_index(query)
        return key, value

    def get_key_and_value_from_index(self, query):
        # logger.debug("query size: %s", query.shape)
        # logger.debug("query.shape[0]: ", query.shape[0])
        # logger.debug("query.shape[1]: ", query.shape[1])
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

    def create_mome_weights(
        self,
        max_loras: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.mome_config = mome_config

        lora_a_out_size = mome_config.max_mome_rank
        lora_b_out_size = self.hidden_size
        self.lora_a_tensors = torch.zeros(
            (                
                max_loras,
                lora_a_out_size,
                lora_b_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )
        self.lora_b_tensors = torch.zeros(
            (
                max_loras,
                lora_b_out_size,
                lora_a_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )

    def reset_mome(self, index: int):
        self.lora_a_tensors[index] = 0
        self.lora_b_tensors[index] = 0

    def set_mome(
        self,
        index: int,
        module_mome: MoMELayerWeights,
    ):
        assert (len(self.lora_a_tensors) == len(self.lora_b_tensors))
        lora_a = module_mome.lora_a
        lora_b = module_mome.lora_b
        self.reset_mome(index)
        self.lora_a_tensors[index, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                   lora_a.T, non_blocking=True)
        self.lora_b_tensors[index, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                   lora_b.T, non_blocking=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(hidden_states)
        logger.info("original mlp output shape %s:", output.shape)
        mome_in_results = F.linear(hidden_states, self.lora_a_tensors[0], bias=None)
        mome_out_results = F.linear(mome_in_results, self.lora_b_tensors[0], bias=None)
        logger.info("mome mlp output shape: %s:", mome_out_results.shape)
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
    def __init__(self, base_layer: ParallelLMHead):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = self.base_layer.embedding_dim
        self.linear_method = getattr(self.base_layer, "linear_method", None)
        if self.linear_method is None:
            raise ValueError(
                "LoraHeadAdaptor init ERROR. The linear_method is not set in the base layer."
            )   
        self.device = _get_mome_device(self.base_layer)

        # mapping tensors
        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor
        self.sampler_indices_gpu: torch.Tensor
        self.indices_len: List[int] = []

        self.head_lora_in = []
        self.head_lora_out = []

    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias
    
    def create_mome_weights(
        self,
        max_loras: int,
        mome_config: MoMEConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.mome_config = mome_config

        lora_a_out_size = mome_config.max_mome_rank
        lora_b_out_size = self.hidden_size
        self.lora_a_tensors = torch.zeros(
            (                
                max_loras,
                lora_a_out_size,
                lora_b_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )
        self.lora_b_tensors = torch.zeros(
            (
                max_loras,
                lora_b_out_size,
                lora_a_out_size,
            ),
            dtype=mome_config.mome_dtype,
            device=self.device,
        )

        self.head_lora_in = [None for _ in range(max_loras)]
        self.head_lora_out = [None for _ in range(max_loras)]

    def _reset_parameters(self, index):
        self.head_lora_out[index].weight.data.zero_()
    
    def reset_mome(self, index: int):
        self.head_lora_in[index] = None
        self.head_lora_out[index] = None

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
        self.head_lora_in[index] = nn.Linear(self.hidden_size, rank, bias=False, device=self.device)
        self.head_lora_out[index] = nn.Linear(rank, self.hidden_size, bias=False, device=self.device)
        self._reset_parameters(index)

    # Call layer with all inputs and kwargs
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(hidden_states)
        mome_in_results = self.head_lora_in[0](hidden_states)
        mome_out_results = self.head_lora_out[0](mome_in_results)
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
