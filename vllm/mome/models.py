from vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from torch import nn
class MoMEModel(AdapterModel):
    """A MoME tuned model."""
    def __init__(self,
                 mome_model_id: str,
                 ) -> None:
        """
        Args:
            mome_model_id: the id (model name) of the MoME model.
        """
        self.id = mome_model_id

    @classmethod
    def from_local_checkpoint(cls,
                              mome_model_id: str,
                              ) -> "MoMEModel":
        """Create a MoMEModel from a local checkpoint.

        Args:
            mome_model_id: the id (model name) of the MoME model.
        """
        return cls(mome_model_id)

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
