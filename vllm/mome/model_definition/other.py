import copy
import torch
from typing import Tuple
from safetensors.torch import storage_ptr, storage_size


def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    This method is the exact same copy of
    https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L282C1-L300C58 but we added
    it here manually to avoid import issue with old versions of transformers.
    """
    unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


def infer_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    return device


def find_mismatched_keys(
    model: torch.nn.Module,
    peft_model_state_dict: dict[str, torch.Tensor],
    ignore_mismatched_sizes: bool = False,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    if not ignore_mismatched_sizes:
        return peft_model_state_dict, []

    mismatched = []
    state_dict = model.state_dict()
    for key, tensor in peft_model_state_dict.items():
        if key not in state_dict:
            continue

        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L3858-L3864
        if (state_dict[key].shape[-1] == 1) and (
            state_dict[key].numel() * 2 == tensor.numel()
        ):
            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size
            # differences. Without matching with module type or paramter type it seems like a practical way to detect
            # valid 4bit weights.
            continue

        if state_dict[key].shape != tensor.shape:
            mismatched.append((key, tensor.shape, state_dict[key].shape))

    for key, _, _ in mismatched:
        del peft_model_state_dict[key]

    return peft_model_state_dict, mismatched


def clone_module(module):
    """Make a shallow copy of a module recursively."""
    shallow_copy = copy.copy(module)
    shallow_copy._parameters = shallow_copy._parameters.copy()
    shallow_copy._buffers = shallow_copy._buffers.copy()
    shallow_copy._modules = shallow_copy._modules.copy()

    for child_name, child in shallow_copy.named_children():
        shallow_copy._modules[child_name] = clone_module(child)
    return shallow_copy
