from vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from torch import nn
from typing import Any, Dict, Optional
from vllm.mome.model_definition.pretrained_lamini_mome_for_causal_lm import load_mome_model_for_inference
    load_mome_model_for_inference

class MoMEModel(AdapterModel):
    """A MoME tuned model."""
    def __init__(self,
                 mome_model_id: str,
                 adapter_path: str,
                 ) -> None:
        """
        Args:
            mome_model_id: the id (model name) of the MoME model.
            adapter_path: the path to the adapter weights.
        """
        self.id = mome_model_id
        self.adapter_path = adapter_path

    @classmethod
    def from_local_checkpoint(cls,
                              mome_model_id: str,
                              adapter_path: str,
                              ) -> "MoMEModel":
        """Create a MoMEModel from a local checkpoint.

        Args:
            mome_model_id: the id (model name) of the MoME model.
            adapter_path: the path to the adapter weights.
        """
        return cls(mome_model_id, adapter_path)

class MoMEModelManager(AdapterModelManager):
    """A manager that manages multiple MoME tuned models."""
    def __init__(self,
                 model: nn.Module,
                 ) -> None:
        """Create a MoMEModel and adapter for a given model.

        Args:
            model: the model to be adapted.
        """
        self.model = model
