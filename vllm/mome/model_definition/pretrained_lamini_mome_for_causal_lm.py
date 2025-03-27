import logging
import os
from typing import Optional, Tuple, Union

import torch
from vllm.mome.model_definition.constants import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from vllm.mome.model_definition.lamini_index import LaminiIndex
from vllm.mome.model_definition.mome_adaptor import (
    add_extra_lora_adapters_to_head,
    add_lora_adaptors_to_mlp_layer,
    add_mome_adaptors_to_each_layer,
)
from vllm.mome.model_definition.mome_config import MoMEConfig
from vllm.mome.model_definition.mome_model_state_dict import (
    is_mome_adapter_layer,
    is_tiny_lm_head_layer,
)
from vllm.mome.model_definition.other import (
    clone_module,
    find_mismatched_keys,
    infer_device,
)
from safetensors.torch import load_file as safe_load_file
from transformers import MistralConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import PushToHubMixin

logger = logging.getLogger(__name__)


def load_mome_model_for_inference(base_model, path):
    logger.debug("Loading MoME model for inference from path: " + path)
    model = PretrainedLaminiMoMEForCausalLM.from_pretrained(
        base_model, os.path.abspath(path)
    )
    model.device = base_model.device

    prepare_mome_model_for_inference(base_model, model)

    return model


def prepare_mome_model_for_inference(base_model, model):
    model.generation_config = base_model.generation_config
    model.config = base_model.config
    model._validate_model_class = base_model._validate_model_class
    model._validate_model_kwargs = base_model._validate_model_kwargs
    model._prepare_model_inputs = base_model._prepare_model_inputs
    model._prepare_attention_mask_for_generation = (
        base_model._prepare_attention_mask_for_generation
    )
    model._validate_generated_length = base_model._validate_generated_length
    model._extract_past_from_model_output = base_model._extract_past_from_model_output
    if hasattr(model, "_get_generation_mode"):
        model._get_generation_mode = base_model._get_generation_mode
    model._get_logits_processor = base_model._get_logits_processor
    model._get_stopping_criteria = base_model._get_stopping_criteria
    model.prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
    model._update_model_kwargs_for_generation = (
        base_model._update_model_kwargs_for_generation
    )
    model._get_initial_cache_position = base_model._get_initial_cache_position
    model._supports_default_dynamic_cache = base_model._supports_default_dynamic_cache
    logger.info(f"Loaded MoME model: {model}")

    return model


class PretrainedLaminiMoMEForCausalLM(PushToHubMixin, torch.nn.Module):
    config_class = MistralConfig
    base_model_prefix = "mome_model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, base_model: PreTrainedModel, config: MoMEConfig):
        super().__init__()
        logger.debug(
            f"PretrainedLaminiMoMEForCausalLM name_or_path {base_model.name_or_path}"
        )
        logger.debug(f"PretrainedLaminiMoMEForCausalLM config {config} {config.path}")
        index_path = os.path.join(config.path, "..", "index")
        self.index = LaminiIndex.load_index(index_path, config.path, cache_dir="cache")

        self.embeddings = {
            "key_embeddings": [],
            "value_embeddings": [],
            "embedding_indices": [],
        }

        cloned_model = clone_module(base_model)
        self.mome_model = add_mome_adaptors_to_each_layer(
            cloned_model, config, self.embeddings, self.index
        )
        self.mome_model = add_lora_adaptors_to_mlp_layer(
            self.mome_model,
            config,
        )
        self.mome_model = add_extra_lora_adapters_to_head(
            base_model.name_or_path, self.mome_model, config
        )
        self.load_adapter(config.path, cloned_model.name_or_path)

    @classmethod
    def from_pretrained(
        cls,
        model,
        model_id,
    ):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """

        # load the config
        config = MoMEConfig.from_pretrained(model_id)
        model = cls(model, config)
        return model

    def load_adapter(
        self,
        model_id: str,
        base_model_name: str,
        is_trainable: bool = False,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        torch_device = infer_device()

        adapters_weights = load_mome_weights(
            model_id,
            device=torch_device,
        )

        logger.debug("LOADED ADAPTERS WEIGHTS: " + str(adapters_weights))
        logger.debug("BEFORE LOADING ADAPTERS: " + str(self.mome_model.state_dict()))
        # load the weights into the model
        load_result = set_mome_model_state_dict_for_inference(
            self.mome_model,
            adapters_weights,
            base_model_name,
        )
        logger.debug("AFTER LOADING ADAPTERS: " + str(load_result))

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.mome_model.eval()
        return load_result

    def forward(self, *args, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.mome_model.forward(*args, **kwargs)

    def generate(self, input_ids, do_sample, max_new_tokens, return_dict_in_generate):
        logger.debug(
            "input_ids in generate: "
            + str(input_ids)
            + " on device: "
            + str(input_ids.device)
        )
        return self.mome_model.generate(
            input_ids=input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_dict_in_generate,
        )


def load_mome_weights(
    model_id: str,
    device: Optional[str] = None,
) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
    """
    path = model_id

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        raise FileNotFoundError(
            f"Could not find the MoME weights at the path: {path}. Please make sure the path is correct."
        )

    if use_safetensors:
        adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights


def set_mome_model_state_dict_for_training(
    model,
    peft_model_state_dict,
    base_model_name,
):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]):
            The Peft model.
        peft_model_state_dict (`dict`):
            The state dict of the Peft model.
    """
    state_dict = {}
    for k, v in peft_model_state_dict.items():
        if is_mome_adapter_layer(k) or is_tiny_lm_head_layer(base_model_name, k):
            state_dict[k] = v

    state_dict, mismatched_keys = find_mismatched_keys(model, state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        raise ValueError(msg)
    return load_result


def set_mome_model_state_dict_for_inference(
    model,
    peft_model_state_dict,
    base_model_name,
):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]):
            The Peft model.
        peft_model_state_dict (`dict`):
            The state dict of the Peft model.
    """
    state_dict = {}
    for k, v in peft_model_state_dict.items():
        if is_mome_adapter_layer(k) or is_tiny_lm_head_layer(base_model_name, k):
            k = k.split("mome_model.")[1]
            state_dict[k] = v

    state_dict, mismatched_keys = find_mismatched_keys(model, state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        raise ValueError(msg)
    return load_result
