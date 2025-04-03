
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

        print("layer:", dir(layer))
        # Get the hidden size
        hidden_size = get_hidden_size(layer)
        
        # [Lamini] UPDATED HIDDEN SIZE FUNCTION TO WORK WITH VLLM
        device = "cuda"
        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(hidden_size, r_value, bias=False, device=device)
        self.mlp_lora_out = nn.Linear(r_value, hidden_size, bias=False, device=device)

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
        device = "cuda"
        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(self.hidden_size[1], r_value, bias=False, device=device)
        self.mlp_lora_out = nn.Linear(r_value, self.hidden_size[0], bias=False, device=device)
        self.linear_method = getattr(self.layer, "linear_method", None)

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

    @property
    def weight(self):
        return self.layer.weight
    @property
    def bias(self):
        return self.layer.bias

    
# [Lamini] UPDATED HIDDEN SIZE FUNCTION TO WORK WITH VLLM
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
