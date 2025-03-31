# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional

import msgspec

from vllm.adapter_commons.request import AdapterRequest


class MoMERequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """
    Request for a MoME adapter.

    Note that this class should be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized MoME adapters.

    mome_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """
    __metaclass__ = AdapterRequest

    mome_name: str
    mome_int_id: int
    mome_path: str = ""
    mome_local_path: Optional[str] = msgspec.field(default=None)
    long_mome_max_len: Optional[int] = None
    base_model_name: Optional[str] = msgspec.field(default=None)

    def __post_init__(self):
        if self.mome_local_path:
            warnings.warn(
                "The 'mome_local_path' attribute is deprecated "
                "and will be removed in a future version. "
                "Please use 'mome_path' instead.",
                DeprecationWarning,
                stacklevel=2)
            if not self.mome_path:
                self.mome_path = self.mome_local_path or ""

        # Ensure mome_path is not empty
        assert self.mome_path, "mome_path cannot be empty"

    @property
    def adapter_id(self):
        return self.mome_int_id

    @property
    def name(self):
        return self.mome_name

    @property
    def path(self):
        return self.mome_path

    @property
    def local_path(self):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.mome_path

    @local_path.setter
    def local_path(self, value):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        self.mome_path = value

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare MoMERequest
        instances based on mome_name. This allows for identification
        and comparison mome adapter across engines.
        """
        return isinstance(value,
                          self.__class__) and self.mome_name == value.mome_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash MoMERequest instances
        based on mome_name. This ensures that MoMERequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.mome_name)
