# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

import torch

from vllm.adapter_commons.utils import (add_adapter_worker,
                                        apply_adapters_worker,
                                        list_adapters_worker,
                                        set_active_adapters_worker)
from vllm.adapter_commons.worker_manager import AbstractWorkerManager
from vllm.config import MoMEConfig
from vllm.logger import init_logger
from vllm.mome.models import (MoMEModel, MoMEModelManager,
                              LRUCacheMoMEModelManager, create_mome_manager)
from vllm.mome.request import MoMERequest
from vllm.lora.utils import get_adapter_absolute_path

logger = init_logger(__name__)


class WorkerMoMEManager(AbstractWorkerManager):
    """WorkerMoMEManager that manages MoME models on the worker side.

    Every request, the requested MoMEs will be loaded (unless they are already
    loaded), and every other MoME will be unloaded."""

    _manager_cls: Type[MoMEModelManager] = MoMEModelManager

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        mome_config: MoMEConfig,
        device: torch.device,
        mome_model_cls: Type[MoMEModel] = MoMEModel,
        max_position_embeddings: Optional[int] = None,
    ):
        self._mome_model_cls = mome_model_cls
        self._cached_dummy_mome: Union[None, Literal[False], MoMEModel] = False
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mome_config = mome_config
        self.max_position_embeddings = max_position_embeddings
        super().__init__(device)
        # Lazily initialized by create_mome_manager.
        self._adapter_manager: MoMEModelManager

    @contextmanager
    def dummy_mome_cache(self):
        """Use this context manager to reuse the dummy mome model
        to avoid creating it repeatedly."""
        self._cached_dummy_mome = None
        yield
        self._cached_dummy_mome = False

    @property
    def is_enabled(self) -> bool:
        return True

    def create_mome_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        mome_manager = create_mome_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            mome_config=self.mome_config,
            device=self.device,
            mome_manager_cls=self._manager_cls,
        )
        self._adapter_manager = mome_manager
        return mome_manager.model

    def _load_adapter(self, mome_request: MoMERequest) -> MoMEModel:
        try:
            model = self._adapter_manager.model
            supported_mome_modules = model.supported_mome_modules
            mome_path = get_adapter_absolute_path(mome_request.mome_path)

            mome = self._mome_model_cls.from_local_checkpoint(
                mome_path,
                supported_mome_modules,
                mome_model_id=mome_request.mome_int_id,
                dtype=self.mome_config.mome_dtype,
                device="cpu"
                )

        except FileNotFoundError as e:
            # FileNotFoundError should be raised if
            # - No local adapter files found at `mome_request.mome_path`
            # For NotFoundError
            raise ValueError(
                f"Loading mome {mome_request.mome_name} failed: No adapter "
                f"found for {mome_path}") from e
        except Exception as e:
            # For BadRequestError
            raise e

        return mome

    def add_dummy_mome(self, mome_request: MoMERequest, rank: int, index_dim: int) -> bool:
        if mome_request.mome_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_mome, MoMEModel):
            dummy_mome = self._cached_dummy_mome.clone(
                mome_request.mome_int_id)
        else:
            dummy_mome = self._adapter_manager.create_dummy_mome(
                mome_request.mome_int_id, rank, index_dim)
            if self._cached_dummy_mome is None:
                self._cached_dummy_mome = dummy_mome
        return self._adapter_manager.add_adapter(dummy_mome)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        # logger.debug("set_active_adapters called. requests:%s mapping: %s", requests, mapping)
        set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)

    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        apply_adapters_worker(adapter_requests, self.list_adapters,
                              self._adapter_manager.adapter_slots,
                              self.remove_adapter, self.add_adapter)

    def add_adapter(self, adapter_request: Any) -> bool:
        return add_adapter_worker(adapter_request, self.list_adapters,
                                  self._load_adapter,
                                  self._adapter_manager.add_adapter,
                                  self._adapter_manager.activate_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        return list_adapters_worker(self._adapter_manager.list_adapters)


class LRUCacheWorkerMoMEManager(WorkerMoMEManager):
    """WorkerM o MEManager that manages MoME models on the worker side.

    Uses an LRU Cache. Every request, the requested MoMEs will be loaded
    (unless they are already loaded) and least recently used MoMEs will
    be unloaded if the cache is above capacity."""

    _manager_cls: Type[LRUCacheMoMEModelManager] = LRUCacheMoMEModelManager

    def create_mome_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        mome_manager = create_mome_manager(
            model,
            mome_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            mome_config=self.mome_config,
            device=self.device,
        )
        self._adapter_manager = mome_manager
        return mome_manager.model

    def _apply_adapters(self, mome_requests: Set[MoMERequest]) -> None:
        momes_map = {
            mome_request.mome_int_id: mome_request
            for mome_request in mome_requests if mome_request
        }
        if len(momes_map) > self._adapter_manager.mome_slots:
            raise RuntimeError(
                f"Number of requested MoMEs ({len(momes_map)}) is greater "
                "than the number of GPU MoME slots "
                f"({self._adapter_manager.mome_slots}).")
        for mome in momes_map.values():
            self.add_adapter(mome)

    def add_adapter(self, mome_request: MoMERequest) -> bool:
        if mome_request.mome_int_id not in self.list_adapters():
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded mome adapters to very temporarily
            # exceed `--max-cpu-momes`.
            mome = self._load_adapter(mome_request)

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager,
                                  LRUCacheMoMEModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(mome)
        else:
            # If the mome is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(
                mome_request.mome_int_id) is not None
        self._adapter_manager.activate_adapter(mome_request.mome_int_id)
        return loaded
