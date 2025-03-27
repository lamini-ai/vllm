from vllm.mome.model_definition.constants import SAFETENSORS_WEIGHTS_NAME
from vllm.mome.model_definition.lamini_index import LaminiIndex
from vllm.mome.model_definition.lora_mlp_adaptor import (
    LoraHeadAdaptor,
    LoraMLPAdaptor,
    get_hidden_size,
)
from vllm.mome.model_definition.other import clone_module, id_tensor_storage
from vllm.mome.model_definition.mome_config import MoMEConfig
from vllm.mome.model_definition.mome_model_state_dict import (
    get_mome_model_state_dict,
    is_mome_adapter_layer,
    is_tiny_lm_head_layer,
)

from safetensors.torch import save_file as safe_save_file
from transformers import MistralConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

import numpy as np
import torch
import torch.nn as nn

import inspect
import collections
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def add_mome_adaptor(model, index_path, r_value, sequence_length, index_k):
    logger.debug("config: " + str(model.config))
    mome_model = LaminiMoMEForCausalLM(model.config, r_value, sequence_length, index_k)

    mome_model.initialize(model, index_path)
    return mome_model


class LaminiMoMEForCausalLM(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "mome_model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, model_config, r_value, sequence_length, index_k):
        super().__init__(model_config)
        logger.debug(f"LaminiMoMEForCausalLM config {model_config}")
        self.mome_config = MoMEConfig(
            base_model_name_or_path=model_config._name_or_path,
            task_type="CAUSAL_LM",
            mome_type="RAFT",
            r_value=r_value,
            sequence_length=sequence_length,
            index_k=index_k,
        )
        self.post_init()

    def initialize(self, model, index_path):
        cloned_model = clone_module(model)

        self.embeddings = {
            "key_embeddings": [],
            "value_embeddings": [],
            "embedding_indices": [],
        }
        self.index = LaminiIndex.load_index(index_path + "/index", index_path, cache_dir="cache")
        self.config.r_value = self.mome_config.r_value
        self.config.sequence_length = self.mome_config.sequence_length
        self.config.index_k = self.mome_config.index_k
        freeze_all_model_params(cloned_model)

        self.mome_model = add_mome_adaptors_to_each_layer(
            cloned_model,
            self.mome_config,
            self.embeddings,
            self.index,
        )

        self.mome_model = add_lora_adaptors_to_mlp_layer(
            self.mome_model,
            self.mome_config,
        )
        self.mome_model = add_extra_lora_adapters_to_head(
            self.mome_config.base_model_name_or_path, self.mome_model, self.mome_config
        )

        mark_only_adapters_as_trainable(self.mome_model)

        if hasattr(self.mome_model, "enable_input_require_grads"):
            self.mome_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.mome_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.mome_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def save_pretrained(
        self, save_directory: str, state_dict: Optional[Dict] = None, **kwargs
    ):
        self.index.save_index_values(save_directory)
        self._save_pretrained_lora_impl(save_directory, state_dict=state_dict, **kwargs)

    def _save_pretrained_lora_impl(
        self,
        save_directory: str,
        is_main_process: bool = True,
        **kwargs: Any,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        mome_config = self.mome_config
        # save only the trainable weights
        output_state_dict = get_mome_model_state_dict(
            self,
            self.mome_config.base_model_name_or_path,
            state_dict=kwargs.get("state_dict", None),
        )
        output_dir = save_directory
        os.makedirs(output_dir, exist_ok=True)
        if is_main_process:
            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in output_state_dict.items():

                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    output_state_dict[shared_tensor_name] = output_state_dict[
                        shared_tensor_name
                    ].clone()

            total_memory = 0
            for k, tensor in output_state_dict.items():
                logger.debug(
                    f"Saving {k} with size {tensor.nelement() * tensor.element_size()}"
                )
                total_memory += tensor.nelement() * tensor.element_size()
                logger.debug(f"SAVING TENSOR value: {str(tensor)[:100]}")
            logger.debug(
                f"Total memory across {len(output_state_dict)} tensors saved: {total_memory}"
            )

            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )

        # save the config and change the inference mode to `True`
        if mome_config.base_model_name_or_path is None:
            mome_config.base_model_name_or_path = self.base_model.model.__dict__.get(
                "name_or_path", None
            )

        if mome_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=False,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if is_main_process:
            mome_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)


def add_mome_adaptors_to_each_layer(
    model: PreTrainedModel, config: MoMEConfig, embeddings, index
):
    """The MoMEAdaptor wraps and replaces layers if they are attention layers."""
    for name, layer in model.named_modules():
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_self_attn(name, layer, model, config, embeddings, index)
    return model


def add_lora_adaptors_to_mlp_layer(
    model: PreTrainedModel,
    config: MoMEConfig,
):
    """Add LoRa adapters to MLP layers."""
    for name, layer in model.named_modules():
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_mlp(name, layer, model, config)
    return model


def add_extra_lora_adapters_to_head(
    base_model_name: str, model: PreTrainedModel, config: MoMEConfig
):
    """Add LoRa adapters to the head."""
    logger.info("Adding LoRa adapters to the head")
    for name, layer in model.named_modules():
        if is_tiny_lm_head_layer(base_model_name, name):
            logger.info(f"Wrapping layer {name} with LoraHeadAdaptor")
            recursive_setattr(model, name, LoraHeadAdaptor(layer, config.r_value))
    return model


def try_to_update_self_attn(
    name,
    layer,
    model: PreTrainedModel,
    config: MoMEConfig,
    embeddings,
    index,
):
    """Try to wrap the layer with a MoMEAdaptor."""
    if not is_self_attn_layer(layer, name):
        return

    logger.info(f"Wrapping layer {name} with MoMEAdaptor")

    # Wrap the layer with a MoMEAdaptor
    recursive_setattr(
        model,
        name,
        MoMEAdaptor(
            layer,
            embeddings,
            index,
            config.r_value,
            config.sequence_length,
            config.index_k,
            requires_attention_output=get_requires_output_attentions(layer),
        ),
    )


def try_to_update_mlp(name, layer, model: PreTrainedModel, config: MoMEConfig):
    """Try to wrap the layer with a LoraMLPAdaptor."""
    if not is_mlp_layer(name):
        return

    logger.info(f"Wrapping layer {name} with LoraMLPAdaptor")

    # Wrap the layer with a MoMEAdaptor
    recursive_setattr(model, name, LoraMLPAdaptor(layer, config.r_value))


def freeze_all_model_params(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False
        logger.debug(
            "Before Params: " + str(n) + " requires grad: " + str(p.requires_grad)
        )


def mark_only_adapters_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        if is_mome_adapter_layer(n):
            p.requires_grad = True


def is_self_attn_layer(layer, name):
    """Check if it is a huggerface attention layer."""
    name_suffix = name.split(".")[-1]

    # huggingface calls it self_attn for mistral models
    if name_suffix == "self_attn":
        return True

    # huggingface calls it attn for other models
    if name_suffix == "attn":
        return True

    return False


def is_mlp_layer(name):
    """Check if it is a huggerface mlp layer."""
    name_suffix = name.split(".")[-1]

    # huggingface calls it mlp
    if name_suffix == "mlp":
        return True

    return False


def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


class MoMEAdaptor(nn.Module):
    def __init__(
        self,
        layer,
        embeddings,
        index,
        r_value,
        mome_embedding_seq_length,
        index_k,
        requires_attention_output,
    ):
        super().__init__()
        self.layer = layer

        self.requires_attention_output = requires_attention_output

        # Add a mome attention layer
        self.mome_attention = MoMEAttentionLayer(
            hidden_size=get_hidden_size(layer),
            r_value=r_value,
            mome_embedding_seq_length=mome_embedding_seq_length,
            device=get_device(layer),
            embeddings=embeddings,
            index=index,
            index_k=index_k,
        )

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):

        self.update_kwargs(
            kwargs, past_key_value=past_key_value, position_ids=position_ids
        )

        # logger.debug(f"hidden states dtype: {hidden_states.dtype}")
        # if attention_mask is not None:
        #    logger.debug(f"attention mask dtype: {attention_mask.dtype}")

        # if position_ids is not None:
        #    logger.debug(f"position ids dtype: {position_ids.dtype}")

        # if past_key_value is not None:
        #    logger.debug(f"past_key_value dtype: {past_key_value}")

        if self.requires_attention_output:
            output_attentions = True

        layer_outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # project the mome attention output to the same size as the transformer attention output
        mome_attention_output = self.mome_attention(hidden_states)

        # logger.debug(
        #     f"mome_attention_output: {mome_attention_output} {torch.histogram(mome_attention_output, bins=4)}"
        # )
        # logger.debug(
        #     f"self_attention_output: {self_attention_output} {torch.histogram(self_attention_output, bins=4)}"
        # )

        layer_and_adaptor_sum = layer_outputs[0] + mome_attention_output

        # sum the two attentions
        return (layer_and_adaptor_sum,) + layer_outputs[1:]

    def update_kwargs(self, kwargs, past_key_value, position_ids):
        args = inspect.getfullargspec(self.layer.forward).args

        if past_key_value is not None:
            if "past_key_value" in args:
                kwargs["past_key_value"] = past_key_value
            elif "layer_past" in args:
                kwargs["layer_past"] = past_key_value
            else:
                assert False, "Could not figure out how to pass past_key_value"

        if position_ids is not None:
            if "position_ids" in args:
                kwargs["position_ids"] = position_ids


attention_head_count = 8


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
        self.index_dimension = min(
            index.embedding_dimension, hidden_size
        )  # need hidden_size?

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
        self.query_projection_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.query_projection_lora_out = nn.Linear(
            r_value, self.index_dimension, bias=False
        )

        # A linear layer to project the value into the space of the original hidden size
        self.value_projection_lora_in = nn.Linear(
            self.index_dimension, r_value, bias=False
        )
        self.value_projection_lora_out = nn.Linear(r_value, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.value_projection_lora_out.weight.data.zero_()
        self.query_projection_lora_out.weight.data.zero_()

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        query = self.get_query(hidden_states)
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
            attn_mask=attention_mask,
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

    def get_query(self, hidden_states):
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


def get_device(layer):
    return next(layer.parameters()).device


def get_requires_output_attentions(layer):
    if type(layer).__name__.find("SpdaAttention") != -1:
        return True

    return False
