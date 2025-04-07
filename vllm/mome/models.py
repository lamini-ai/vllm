import copy
import re
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

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
from vllm.mome.layers import (AttentionLayerWithMoME, MoMEMapping)
from vllm.mome.utils import (from_layer, replace_submodule)

logger = init_logger(__name__)

_GLOBAL_MOME_ID = 0


def get_mome_id():
    global _GLOBAL_MOME_ID
    _GLOBAL_MOME_ID += 1
    return _GLOBAL_MOME_ID

class MoMEModel(AdapterModel):
    """A MoME tuned model."""
    def __init__(
            self,
            mome_model_id: str,
            rank: int,
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

    def clone(self, mome_model_id: int) -> "MoMEModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            mome_model_id,
            rank=self.rank,
        )

    @classmethod
    def from_local_checkpoint(cls,
                            mome_dir: str,
                            peft_helper: Any,
                            mome_model_id: str,
                            device: str = "cuda",
                            ) -> "MoMEModel":
        """Create a MoMEModel from a local checkpoint.

        Args:
            mome_dir: the directory of the local checkpoint.
            mome_model_id: the id (model name) of the MoME model.
        """
        return None

class MoMEModelManager(AdapterModelManager):
    """A manager that manages multiple MoME tuned models."""
    def __init__(
            self,
            model: nn.Module,
            mome_config: MoMEConfig,
            device: torch.device
            ) -> None:
        """Create a MoMEModel and adapter for a given model.

        Args:
            model: the model to be adapted.
        """
        self.mome_config = mome_config
        self.device = device
        assert self.capacity >= self.mome_slots
        self.mome_index_to_id: List[Optional[int]] = [None] * self.mome_slots

        super().__init__(model)
        self.model = model
        if hasattr(self.model, "supported_mome_modules"):
            self.supported_mome_modules = copy.deepcopy(
                self.model.supported_mome_modules)
        self.modules: Dict[str, Any] = {}
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
        # Implementation for activating an adapter
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        """Deactivate a specific adapter by its ID."""
        # Implementation for deactivating an adapter
        return True

    def _add_adapter(self, mome: MoMEModel):
        pass

    def add_adapter(self, adapter: MoMEModel) -> bool:
        logger.debug(
            "Adding mome. Model id: %d, "
            "int id: %d, "
            "scaling factor: %s", adapter.id, adapter.id,
            adapter.scaling_factor)
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def _set_adapter_mapping(self, mapping: MoMEMapping) -> None:
        pass

    def _create_mome_modules(self):
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            pass
            # if not self._match_target_modules(module_name):
            #     continue
            
            # # 1. Normal MOME injection
            # if "mlp" in module_name:
            #     # add_mome_adaptors_to_mlp_layer
            #     new_module = replace_submodule(
            #         self.model, 
            #         module_name, 
            #         from_layer(module, self.mome_slots, self.mome_config, self.model.config)
            #     )
            
            # # 2. Extra adapter for head
            # elif "lm_head" in module_name:
            #     new_module = replace_submodule(
            #         self.model,
            #         module_name,
            #         from_layer_logits_processor(module, self.mome_slots, self.mome_config, self.model.config)
            #     )
            
            # # 3. Standard MoME adapter
            # else:
            #     new_module = replace_submodule(
            #         self.model,
            #         module_name,
            #         from_layer(module, self.mome_slots, self.mome_config, self.model.config)
            #     )

            # # set index / embeddings
            # if hasattr(new_module, "set_index"):
            #     new_module.set_index(self.lamini_index)
            # if hasattr(new_module, "set_embeddings"):
            #     new_module.set_embeddings(self.embeddings)

            # self.register_module(module_name, new_module)
            # All lora layers share the same punica_wrapper based on reference.
            # new_module.set_mapping(self.punica_wrapper)
    
    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_mome_modules)

    def register_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def set_adapter_mapping(self, mapping: MoMEMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

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