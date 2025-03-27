import logging

from vllm.mome.model_definition.constants import (
    MOME_ADAPTER_PREFIXES,
    MOME_KEY_VALUE_PREFIXES,
)

logger = logging.getLogger(__name__)


def is_mome_adapter_layer(name: str):
    return any(prefix in name for prefix in MOME_ADAPTER_PREFIXES)


def is_mome_key_value(name: str):
    return any(prefix in name for prefix in MOME_KEY_VALUE_PREFIXES)


def is_tiny_lm_head_layer(base_model_name: str, name: str):
    # if "hf-internal-testing" not in base_model_name:
    #    return False
    return any(prefix in name for prefix in ["lm_head"])


def get_mome_model_state_dict(
    model,
    base_model_name: str,
    state_dict=None,
):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
    """

    if state_dict is None:
        state_dict = model.state_dict()
    # to_return = mome_state_dict(model, bias=model.peft_config.bias)
    # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
    # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
    to_return = {
        k: state_dict[k]
        for k in state_dict
        if is_mome_adapter_layer(k) or is_tiny_lm_head_layer(base_model_name, k)
    }
    return to_return
