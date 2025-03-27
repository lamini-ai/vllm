from vllm.adapter_commons.models import AdapterModel

class MoMEModel(AdapterModel):
    """A MoME tuned model."""
    def __init__(self,
                 mome_model_id: str,
                 ) -> None:
        self.id = mome_model_id

    @classmethod
    def from_local_checkpoint(cls,
                              mome_model_id: str,
                              ) -> "MoMEModel":
        return cls(mome_model_id)
