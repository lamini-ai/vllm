import copy
import os
import re
import json
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union, Tuple

import safetensors.torch
import torch
from torch import nn

from typing import Any, Dict, Optional
from vllm.mome.model_definition.pretrained_lamini_mome_for_causal_lm import load_mome_model_for_inference

from vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)

from vllm.config import MoMEConfig
from vllm.logger import init_logger

from vllm.mome.mome import MoMELayerWeights
from vllm.mome.model_definition.lamini_index import LaminiIndex
from vllm.mome.layers import (BaseLayerWithMoME, MoMEAttentionLayer, MoMEMapping)
from vllm.mome.utils import (from_layer, from_layer_logits_processor, replace_submodule, parse_fine_tuned_mome_name)

from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)

_GLOBAL_MOME_ID = 0


def get_mome_id():
    global _GLOBAL_MOME_ID
    _GLOBAL_MOME_ID += 1
    return _GLOBAL_MOME_ID


def convert_mapping(
    mapping: MoMEMapping,
    mome_index_to_id: List[Optional[int]],
    max_momes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], List[int]]:
    """Converts MoMEMapping to index tensors.

    Args:
        mapping: M o MEMapping mapping rows in a batch to MoME ids.
        mome_index_to_id: List mapping MoME ids to MoME indices.
        max_momes: Maximum number of MoMEs.
        device: Device to use for the tensors.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                MoME indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                MoME indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to MoME indices.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the MoMEs, second row is for the MoME.lora_a
                embeddings.
            indices_len: List of lengths of the above tensors. It contains
                (base_indices, sampler_indices, embeddings_indices).
    """
    index_mapping_indices: List[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()

    prompt_mapping: List[int] = [
        mome_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        lora_idx = (mome_index_to_id.index(index_mapping_indices[i])
                    if index_mapping_indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx

    indices_list: List[Union[List[int], torch.Tensor]] = [
        index_mapping_indices,
        lora_indices,
        embedding_indices,
    ]
    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    prompt_mapping_tensor = torch.tensor(prompt_mapping,
                                         dtype=torch.long,
                                         device=device)
    embeddings_indices = indices[2].unsqueeze(0)
    embeddings_indices[embeddings_indices == -1] = max_momes - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor

    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        embeddings_indices.shape[-1],
    ]

    return (
        base_indices,
        sampler_indices,
        embeddings_indices,
        indices_len,
    )

class MoMEModel(AdapterModel):
    """A MoME tuned model."""
    def __init__(
            self,
            mome_model_id: str,
            rank: int,
            momes: Dict[str, MoMELayerWeights],
            indexs: Optional[list[int]] = None,
            ) -> None:
        """
        Args:
            mome_model_id: the id (model name) of the MoME model.
            rank: mome rank.
        """
        self.id = mome_model_id
        assert (
            mome_model_id
            > 0), f"a valid mome id should be greater than 0, got {self.id}"
        self.rank = rank
        self.momes: Dict[str, MoMELayerWeights] = momes

    def clone(self, mome_model_id: int) -> "MoMEModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            mome_model_id,
            rank=self.rank,
            momes=self.momes.copy(),
        )

    def get_mome(self, module_name: str) -> Optional[MoMELayerWeights]:
        """Get MoME for a given module by name"""
        return self.momes.get(module_name, None)

    @classmethod
    def from_mome_tensors(
        cls,
        mome_model_id: str,
        tensors: Dict[str, torch.Tensor],
        mome_index: LaminiIndex,
        config_content: Dict[str, Union[int, str]],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "MoMEModel":
        """Create a MoMEModel from tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        momes: Dict[str, MoMELayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a, is_mome_attention, _, _ = parse_fine_tuned_mome_name(
                tensor_name)
            if module_name not in momes:
                momes[module_name] = MoMELayerWeights.from_config(
                    module_name, config_content["r_value"])

            if is_mome_attention:
                momes[module_name].index = mome_index
                momes[module_name].index_k = config_content["index_k"]

            if is_lora_a:
                momes[module_name].lora_a = tensor.to(device=device,
                                                      dtype=dtype).t()
                if pin_memory:
                    momes[module_name].lora_a = momes[
                        module_name].lora_a.pin_memory()
            else:
                momes[module_name].lora_b = tensor.to(device=device,
                                                      dtype=dtype).t()

        return cls(mome_model_id, config_content["r_value"], momes)

    @classmethod
    def from_local_checkpoint(cls,
                            mome_dir: str,
                            expected_mome_modules: List[str],
                            mome_model_id: str,
                            dtype: Optional[torch.dtype] = None,
                            device: str = "cuda",
                            ) -> "MoMEModel":
        """Create a MoMEModel from a local checkpoint.

        Args:
            mome_dir: the directory of the local checkpoint.
            mome_model_id: the id (model name) of the MoME model.
        """

        mome_tensor_path = os.path.join(mome_dir, "adapter_model.safetensors")
        mome_bin_file_path = os.path.join(mome_dir, "adapter_model.bin")

        index_path = os.path.join(mome_dir, "..", "index")
        mome_index = LaminiIndex.load_index(index_path, mome_dir, cache_dir="cache")

        config_path = os.path.join(mome_dir, "adapter_config.json")
        if os.path.isfile(config_path):
            config_content = json.loads(config_path)
        else:
            raise ValueError(f"{mome_dir} doesn't contain adapter_config.json")

        if os.path.isfile(mome_tensor_path):
            tensors: Dict[str, torch.Tensor] = {}
            unexpected_modules = []
            with safetensors.safe_open(mome_tensor_path,
                                       framework="pt") as f:  # type: ignore
                for mome_module in f.keys():  # noqa
                    module_name, _, _, _, _ = parse_fine_tuned_mome_name(mome_module)
                    part_name = module_name.split(".")[-1]
                    # here part_name should be one of ["mome_attention", "mlp", "lm_head"]
                    if part_name not in expected_mome_modules:
                        unexpected_modules.append(module_name)
                if unexpected_modules:
                    raise ValueError(
                        f"While loading {mome_dir}, expected"
                        f" target modules in {expected_mome_modules}"
                        f" but received {unexpected_modules}."
                        f" Please verify that the loaded MoME module is correct"
                    )
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(mome_bin_file_path):
            unexpected_modules = []
            tensors = torch.load(mome_bin_file_path, map_location=device)
        else:
            raise ValueError(f"{mome_dir} doesn't contain tensors")

        return cls.from_mome_tensors(
            mome_model_id=get_mome_id()
            if mome_model_id is None else mome_model_id,
            tensors=tensors,
            mome_index=mome_index,
            config_content=config_content,
            dtype=dtype,
            device=device)

class MoMEModelManager(AdapterModelManager):
    """A manager that manages multiple MoME tuned models."""
    def __init__(
            self,
            model: nn.Module,
            max_num_seqs: int,
            max_num_batched_tokens: int,
            mome_config: MoMEConfig,
            device: torch.device
            ) -> None:
        """Create a MoMEModel and adapter for a given model.

        Args:
            model: the model to be adapted.
        """
        self.mome_config = mome_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.mome_slots
        self.mome_index_to_id: List[Optional[int]] = [None] * self.mome_slots
        # self.punica_wrapper = get_punica_wrapper(max_num_batched_tokens,
        #                                          max_batches=self.max_num_seqs,
        #                                          device=self.device)
        # Use the simple wrapper for now
        self.base_indices = torch.tensor([-1])
        self.sampler_indices = torch.tensor([-1])
        self.base_embedding_indices = torch.tensor([])
        self.indices_len = [0, 0, 0]

        super().__init__(model)
        self.model = model
        if hasattr(self.model, "supported_mome_modules"):
            self.supported_mome_modules = copy.deepcopy(
                self.model.supported_mome_modules)
            self.packed_modules_mapping = copy.deepcopy(
                self.model.packed_modules_mapping)
        self.modules: Dict[str, BaseLayerWithMoME] = {}
        self._last_mapping: Optional[MoMEMapping] = None
        self._create_mome_modules()
        self.model.mome_manager = self
        self.adapter_type = 'MoME'

    @property
    def capacity(self) -> int:
        return self.mome_config.max_cpu_momes

    @property
    def mome_slots(self) -> int:
        return self.mome_config.max_momes

    @property
    def adapter_slots(self) -> int:
        return self.mome_slots

    def activate_adapter(
        self,
        mome_id: int,
    ) -> bool:
        """Activate a specific adapter by its ID."""
        if mome_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, mome_id) for i, mome_id in enumerate(self.mome_index_to_id)
             if mome_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free mome slots")
        index, _ = first_free_slot
        self._active_adapters[mome_id] = None
        mome_model = self._registered_adapters[mome_id]
        logger.info("Activating MoME. int id: %d, slot index: %d",
                     mome_model.id, index)
        self.mome_index_to_id[index] = mome_model.id
        for module_name, module in self.modules.items():
            module_mome = mome_model.get_mome(module_name)
            if module_mome:
                module.set_mome(index, module_mome.lora_a,
                                 module_mome.lora_b, module_mome.rank,
                                 module_mome.index, module_mome.index_k)
            else:
                module.reset_mome(index)
        return True

    def _deactivate_adapter(self, momo_id: int) -> bool:
        """Deactivate a specific adapter by its ID."""
        try:
            index = self.mome_index_to_id.index(momo_id)
            self.mome_index_to_id[index] = None
        except ValueError:
            pass

    def _add_adapter(self, mome: MoMEModel):
        self._registered_adapters[mome.id] = mome

    def add_adapter(self, adapter: MoMEModel) -> bool:
        logger.debug(
            "Adding mome. Model id: %d, "
            "int id: %d, ", adapter.id, adapter.id)
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def _set_adapter_mapping(self, mapping: MoMEMapping) -> None:
        (
            base_indices,
            sampler_indices,
            embeddings_indices,
            indices_len
        ) = convert_mapping(
            mapping,  self.mome_index_to_id, self.mome_slots + 1, self.device)
        for _, module in self.modules.items():
            module.set_mapping(base_indices, sampler_indices, embeddings_indices, indices_len)

    def _create_mome_modules(self):
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue
            if not self._match_target_modules(module_name):
                continue 
            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            # 1. MLP LoRA injection
            if "mlp" in module_name:
                # add_mome_adaptors_to_mlp_layer
                new_module = replace_submodule(
                    self.model, 
                    module_name, 
                    from_layer(module, self.mome_slots, self.mome_config, packed_moduled_lst, self.model.config)
                )
            # 2. Extra LoRA for head
            elif "lm_head" in module_name:
                new_module = replace_submodule(
                    self.model,
                    module_name,
                    from_layer_logits_processor(module, self.mome_slots, self.mome_config, packed_moduled_lst, self.model.config)
                )
            # 3. Standard MoME adapter
            else:
                new_module = replace_submodule(
                    self.model,
                    module_name,
                    from_layer(module, self.mome_slots, self.mome_config, packed_moduled_lst, self.model.config)
                )
            self.register_module(module_name, new_module)
            # TODO All mome layers share the same punica_wrapper based on reference.
            new_module.set_mapping(self.base_indices,
                                   self.sampler_indices,
                                   self.base_embedding_indices,
                                   self.indices_len)

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_mome_modules)

    def register_module(self, module_name: str, module: "BaseLayerWithMoME"):
        assert isinstance(module, BaseLayerWithMoME)
        self.modules[module_name] = module

    def set_adapter_mapping(self, mapping: MoMEMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def remove_all_adapters(self):
        """Remove all MoMEModels from the manager."""
        self._registered_adapters.clear()
        self.mome_index_to_id = [None] * self.mome_slots
        self._active_adapters.clear()

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def pin_adapter(self, mome_id: int) -> bool:
        """Pin a MoMEModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in MoMEModelManager.")


def create_mome_manager(
        model: nn.Module,
        mome_config: MoMEConfig,
        device: torch.device,
        mome_manager_cls: Type[MoMEModelManager] = MoMEModelManager,
        **kwargs) -> MoMEModelManager:
    """Create a MoME adapter for a given model."""
    logger.warning(f"Make Sure Model {type(model)} is supported for MoME.")
    mome_manager = mome_manager_cls(
        model=model,
        mome_config=mome_config,
        device=device,
        **kwargs)
    return mome_manager
