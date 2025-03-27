import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)


class LoraMLPAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer

        # Get the hidden size
        hidden_size = get_hidden_size(layer)

        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

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
        base_model_results = self.layer(hidden_states)
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)
        # logger.debug(
        #     f"lora_results: {lora_results} {torch.histogram(lora_results, bins=4)}"
        # )
        # logger.debug(
        #     f"base_model_results: {base_model_results} {torch.histogram(base_model_results, bins=4)}"
        # )
        # sum the two outputs
        layer_and_adaptor_sum = base_model_results + lora_results

        return layer_and_adaptor_sum


class LoraHeadAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer
        self.hidden_size = layer.weight.shape
        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(self.hidden_size[1], r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, self.hidden_size[0], bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

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
        results = self.layer(hidden_states)
        # logger.debug(
        #     f"lora_results: {lora_results} {torch.histogram(lora_results, bins=4)}"
        # )
        # logger.debug(
        #     f"base_model_results: {base_model_results} {torch.histogram(base_model_results, bins=4)}"
        # )
        # sum the two outputs
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)

        return results + lora_results


def get_hidden_size(layer):
    logger.debug(f"getting hidden size for layer: {layer}")
    if hasattr(layer, "attention"):
        return get_hidden_size(layer.attention)

    if hasattr(layer, "hidden_size"):
        logger.debug(f"hidden size: {layer.hidden_size} from layer.hidden_size")
        return layer.hidden_size

    if hasattr(layer, "q_proj"):
        logger.debug(f"hidden size: {layer.q_proj.weight.shape[1]} from layer.q_proj")
        return layer.q_proj.weight.shape[1]

    if hasattr(layer, "out_proj"):
        logger.debug(
            f"hidden size: {layer.out_proj.weight.shape[1]} from layer.out_proj"
        )
        return layer.out_proj.weight.shape[1]

    if hasattr(layer, "c_fc"):
        logger.debug(f"hidden size: {layer.c_fc.weight.shape[1]} from layer.c_fc")
        return layer.c_fc.weight.shape[1]

    if hasattr(layer, "fc2"):
        logger.debug(f"hidden size: {layer.fc2.weight.shape[0]} from layer.fc2")
        return layer.fc2.weight.shape[0]

    if hasattr(layer, "gate_up_proj"):
        logger.debug(
            f"hidden size: {layer.gate_up_proj.weight.shape[1]} from layer.gate_up_proj"
        )
        return layer.gate_up_proj.weight.shape[1]

    assert hasattr(layer, "c_proj")
    logger.debug(f"hidden size: {layer.c_proj.weight.shape[1]} from layer.c_proj")
    return layer.c_proj.weight.shape[1]
