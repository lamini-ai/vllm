# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

from vllm.mome.models import MoMEModel
from vllm.model_executor.models.llama import LlamaForCausalLM

def test_load_checkpoints():
    supported_mome_modules = LlamaForCausalLM.supported_mome_modules
    packed_modules_mapping = LlamaForCausalLM.packed_modules_mapping
    expected_mome_modules: List[str] = []
    for module in supported_mome_modules:
        if module in packed_modules_mapping:
            expected_mome_modules.extend(packed_modules_mapping[module])
        else:
            expected_mome_modules.append(module)

    MoMEModel.from_local_checkpoint(
        "/root/34916/checkpoints/checkpoint-60",
        expected_mome_modules,
        mome_model_id=1,
        device="cpu")
