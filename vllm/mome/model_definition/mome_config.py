import enum
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union

from vllm.mome.model_definition.constants import CONFIG_NAME
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

logger = logging.getLogger(__name__)


class MoMEType(str, enum.Enum):
    MOME = "MOME"


class TaskType(str, enum.Enum):
    CAUSAL_LM = "CAUSAL_LM"


@dataclass
class MoMEConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for MOME adapter models. It contains all the methods that are common to all
    MOME adapter models. This class inherits from [`~transformers.utils.PushToHubMixin`] which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    Args:
        mome_type (Union[[`~mome.utils.config.MoMEType`], `str`]): The type of MoME method to use.
    """

    mome_type: Optional[MoMEType] = field(
        default=None, metadata={"help": "The type of MOME model."}
    )

    def to_dict(self) -> Dict:
        r"""
        Returns the configuration for your adapter model as a dictionary.
        """
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = asdict(self)
        # converting set type to list
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_mome_type(cls, **kwargs):
        config_cls = cls
        return config_cls(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            raise ValueError(
                f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'"
            )

        loaded_attributes = cls.from_json_file(config_file)
        kwargs = {**class_kwargs, **loaded_attributes, "path": path}
        return cls.from_mome_type(**kwargs)

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file) as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def _get_mome_type(
        cls,
        model_id: str,
        **hf_hub_download_kwargs,
    ):
        subfolder = hf_hub_download_kwargs.get("subfolder", None)

        path = os.path.join(model_id, subfolder) if subfolder is not None else model_id

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    model_id,
                    CONFIG_NAME,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{model_id}'")

        loaded_attributes = cls.from_json_file(config_file)
        return loaded_attributes["mome_type"]

    @property
    def is_prompt_learning(self) -> bool:
        r"""
        Utility method to check if the configuration is for prompt learning.
        """
        return False

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return False


@dataclass
class MoMEConfig(MoMEConfigMixin):

    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    mome_type: Optional[Union[str, MoMEType]] = field(
        default=None, metadata={"help": "MoME type"}
    )
    task_type: Optional[Union[str, TaskType]] = field(
        default=None, metadata={"help": "Task type"}
    )
    path: Optional[str] = field(
        default=None, metadata={"help": "The path to the adapter and index."}
    )
    r_value: Optional[int] = field(
        default=None, metadata={"help": "Inner dimension of lora adapter."}
    )
    sequence_length: Optional[int] = field(
        default=None, metadata={"help": "The block size to train over."}
    )
    index_k: Optional[int] = field(
        default=None, metadata={"help": "Number of nearest neighbors to consider."}
    )


def is_mome_config(adapter_config_file):
    logger.debug(f"is_mome_config {adapter_config_file}")
    try:
        _ = MoMEConfig.from_pretrained(adapter_config_file)
        logger.debug(f"is_mome_config true ")

        return True
    except Exception as e:
        logger.debug(e)
        logger.debug(f"is_mome_config exception ")

        return False
