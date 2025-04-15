# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import List, Optional, Set, Tuple, Type, Union, Any

import huggingface_hub
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import MoMEConfig
from vllm.logger import init_logger
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable

from vllm.mome.layers import (BaseLayerWithMoME, BaseMoMEAttentionLayer,
                              LoraMLPAdaptor, LoraHeadAdaptor)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_all_mome_classes: Set[Type[BaseLayerWithMoME]] = {
    BaseMoMEAttentionLayer,
    LoraMLPAdaptor,
    LoraHeadAdaptor
}


def from_layer(layer: nn.Module,
               max_momes: int,
               mome_config: MoMEConfig,
               packed_modules_list: List,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for mome_cls in _all_mome_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if mome_cls.can_replace_layer(source_layer=layer,
                                      mome_config=mome_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            ret = mome_cls(layer)
            ret.create_mome_weights(max_momes, mome_config, model_config)
            return ret
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_momes: int,
    mome_config: MoMEConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LoraHeadAdaptor:
    # TODO: update LoraHeadAdaptor init to work here
    ret = LoraHeadAdaptor(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_mome_weights(max_momes, mome_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


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


def parse_fine_tuned_mome_name(
        name: str,
        weights_mapper: Optional[WeightsMapper] = None
) -> Tuple[str, bool, bool]:
    """Parse the name of mome weights.

    args:
        name: the name of the fine-tuned MoMe, e.g.
            mome_model.lm_head.layer.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
        Tuple(module_name, is_lora_a, is_mome_attention, is_mlp_lora, is_head_lora):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is input layer or output layer.
            is_mome_attention whether the tensor is mome_attention lora .
            is_mlp_lora whether the tensor is mlp_lora.
            is_head_lora whether the tensor is head_lora.
    """

    parts = name.split(".")
    is_mome_attention = False
    is_mlp_lora = False
    is_head_lora = False

    if parts[-1] == "weight":
        if parts[-3] == "mome_attention"
            is_mome_attention = True
        elif parts[-3] == "mlp":
            is_mlp_lora = True
        elif parts[-3] == "lm_head":
            is_head_lora = True
        new_name = ".".join(parts[1:-2])
        return new_name, "in" in parts[-2], is_mome_attention, is_mlp_lora, is_head_lora

    raise ValueError(f"{name} is unsupported MoME weight")


def is_regex_target_modules(load_modules: Union[str, List[str]],
                            expected_lora_modules: List[str]) -> bool:
    """
    PEFT supports passing `target_modules` in the form of regular expressions,
    such as `model.*(q_proj|k_proj|v_proj)$`. This function is mainly used to
    determine whether the suffix in the regular expression is present in the
    `expected_lora_modules`.
    """

    def is_valid_regex(pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def is_subset(sub_list, full_list):
        return set(sub_list).issubset(set(full_list))

    # Similar to PEFT's processing logic, regex-related operations are only
    #  executed when the load_modules is a `str`.
    if not isinstance(load_modules, str):
        return False

    if is_valid_regex(load_modules):
        match = re.search(r"\((.*?)\)\$?$", load_modules)
        if match:
            suffix = match.group(1).split("|")
            return is_subset(suffix, expected_lora_modules)
    return False


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.

    If the lora_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the lora_path as a local file path and
    converts it to an absolute path.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the lora model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith('~'):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(
            repo_id=lora_path)
    except (HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError,
            HFValidationError):
        # Handle errors that may occur during the download
        # Return original path instead instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return lora_path

    return local_snapshot_path
