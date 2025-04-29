import os
import json
import logging
import numpy as np
import torch
# import faiss

from vllm.logger import init_logger

logger = init_logger(__name__)


class LaminiIndex:
    def __init__(
        self,
        device: str,
    ):
        self.device = device
        self.index = None
        self.splits = None
        self.keys = None
        self.values = None
        self.embedding_dimension = None

    @staticmethod
    def load_index(key_path: str, values_path: str, device: str = "cuda") -> "LaminiIndex":
        lamini_index = LaminiIndex(device)

        '''
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        config_path = os.path.join(path, "index_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            lamini_index.embedding_dimension = config["embedding_dimension"]

        # Load the index
        lamini_index.index = faiss.read_index(faiss_path)

        # Load splits
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)
        '''

        # Load keys
        keys_path_json = os.path.join(key_path, "keys.json")
        keys_path_npy = os.path.join(key_path, "keys.npy")
        if os.path.exists(keys_path_json):
            with open(keys_path_json, "r") as f:
                lamini_index.keys = torch.tensor(json.load(f), dtype=torch.float32)
        elif os.path.exists(keys_path_npy):
            lamini_index.keys = torch.from_numpy(np.load(keys_path_npy)).float()
        else:
            raise ValueError("Keys file not found")

        # Load values
        values_path_json = os.path.join(values_path, "values.json")
        values_path_npy = os.path.join(values_path, "values.npy")
        if os.path.exists(values_path_json):
            with open(values_path_json, "r") as f:
                lamini_index.values = torch.tensor(json.load(f), dtype=torch.float32)
        elif os.path.exists(values_path_npy):
            lamini_index.values = torch.from_numpy(np.load(values_path_npy)).float()
        else:
            raise ValueError("Values file not found")

        lamini_index.keys = lamini_index.keys.to(device)
        lamini_index.values = lamini_index.values.to(device)

        return lamini_index
    
    def get_key_and_value(self, query_embeddings: torch.Tensor, k: int) -> tuple:
        # logger.debug(f"query_embeddings shape: {query_embeddings.shape}")
        device = query_embeddings.device
        dtype = query_embeddings.dtype

        query_norm = torch.nn.functional.normalize(query_embeddings.float(), dim=-1)
        keys_norm = torch.nn.functional.normalize(self.keys.float(), dim=-1)

        similarities = torch.matmul(query_norm, keys_norm.T)  # [B, N]

        topk_values, topk_indices = similarities.topk(k, dim=-1)  # [B, k]

        flat_indices = topk_indices.view(-1)

        selected_keys = self.keys.index_select(0, flat_indices)
        # selected_keys = selected_keys.view(topk_indices.shape[0], topk_indices.shape[1], -1)
        # logger.debug(f"selected_keys shape: dtype: ", selected_keys.shape, selected_keys.dtype)
        selected_values = self.values.index_select(0, flat_indices)
        # selected_values = selected_values.view(topk_indices.shape[0], topk_indices.shape[1], -1)
        # logger.debug(f"selected_values shape: dtype: ", selected_values.shape, selected_values.dtype)

        return selected_keys.to(device=device, dtype=dtype), selected_values.to(device=device, dtype=dtype), topk_indices
