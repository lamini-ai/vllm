import os
import json
import numpy as np
import torch

from typing import Optional

from vllm.utils import is_pin_memory_available
from vllm.logger import init_logger

logger = init_logger(__name__)


class LaminiIndex:
    def __init__(
        self,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.keys = None
        self.values = None
        self.embedding_dimension = None

    @staticmethod
    def load_index(key_path: str, values_path: str, 
                   dtype: Optional[torch.dtype] = None, device: str = "cuda", ) -> "LaminiIndex":
        logger.debug(f"Loading LaminiIndex from {key_path} and {values_path} with dtype {dtype} and device {device}")
        lamini_index = LaminiIndex(device)

        # Load keys
        keys_path_json = os.path.join(key_path, "keys.json")
        keys_path_npy = os.path.join(key_path, "keys.npy")
        if os.path.exists(keys_path_json):
            with open(keys_path_json, "r") as f:
                lamini_index.keys = torch.tensor(json.load(f), dtype=dtype)
        elif os.path.exists(keys_path_npy):
            lamini_index.keys = torch.from_numpy(np.load(keys_path_npy)).to(dtype)
        else:
            raise ValueError("Keys file not found")

        # Load values
        values_path_json = os.path.join(values_path, "values.json")
        values_path_npy = os.path.join(values_path, "values.npy")
        if os.path.exists(values_path_json):
            with open(values_path_json, "r") as f:
                lamini_index.values = torch.tensor(json.load(f), dtype=dtype)
        elif os.path.exists(values_path_npy):
            lamini_index.values = torch.from_numpy(np.load(values_path_npy)).to(dtype)
        else:
            raise ValueError("Values file not found")
        
        lamini_index.keys = lamini_index.keys.to(device)
        lamini_index.values = lamini_index.values.to(device)
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        if pin_memory:
            lamini_index.keys = lamini_index.keys.pin_memory()
            lamini_index.values = lamini_index.values.pin_memory()

        return lamini_index
    
    def get_key_and_value(self, query_embeddings: torch.Tensor, k: int) -> tuple:
        # logger.debug(f"query_embeddings shape: {query_embeddings.shape}")

        query_norm = torch.nn.functional.normalize(query_embeddings, dim=-1)
        keys_norm = torch.nn.functional.normalize(self.keys, dim=-1)

        similarities = torch.matmul(query_norm, keys_norm.T)  # [B, N]

        topk_values, topk_indices = similarities.topk(k, dim=-1)  # [B, k]

        flat_indices = topk_indices.view(-1)

        selected_keys = self.keys.index_select(0, flat_indices)
        # selected_keys = selected_keys.view(topk_indices.shape[0], topk_indices.shape[1], -1)
        # logger.debug(f"selected_keys shape: dtype: ", selected_keys.shape, selected_keys.dtype)
        selected_values = self.values.index_select(0, flat_indices)
        # selected_values = selected_values.view(topk_indices.shape[0], topk_indices.shape[1], -1)
        # logger.debug(f"selected_values shape: dtype: ", selected_values.shape, selected_values.dtype)
        return selected_keys, selected_values, topk_indices

    @staticmethod
    def dummy_index(embedding_dimension: int, dtype: Optional[torch.dtype] = None,
                    device: str = "cuda", num_entries: int = 1024) -> "LaminiIndex":
        """Create a dummy index with random keys and values for testing."""
        lamini_index = LaminiIndex(device=device, dtype=dtype)
        lamini_index.embedding_dimension = embedding_dimension

        # keys: [num_entries, embedding_dim]
        lamini_index.keys = torch.randn(num_entries, embedding_dimension, dtype=dtype, device=device)
        lamini_index.values = torch.randn(num_entries, embedding_dimension, dtype=dtype, device=device)

        return lamini_index