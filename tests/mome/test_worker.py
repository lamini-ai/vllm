# SPDX-License-Identifier: Apache-2.0

import os
import random
import tempfile
from unittest.mock import patch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, MoMEConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.mome.models import MoMEMapping
from vllm.mome.request import MoMERequest
from vllm.worker.worker import Worker


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_mome(mome_adapter_files):
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "meta-llama/Llama-3.1-8B-Instruct",
            task="auto",
            tokenizer="meta-llama/Llama-3.1-8B-Instruct",
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
            revision=None,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(1, 1, False),
        scheduler_config=SchedulerConfig("generate", 32, 32, 32),
        device_config=DeviceConfig("cuda"),
        cache_config=CacheConfig(block_size=16,
                                 gpu_memory_utilization=1.,
                                 swap_space=0,
                                 cache_dtype="auto"),
        mome_config=MoMEConfig(max_mome_rank=8, max_cpu_momes=32,
                               max_momes=32),
    )
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )
    worker.init_device()
    worker.load_model()

    worker.model_runner.set_active_momes([], MoMEMapping([], []))
    assert worker.list_momes() == set()

    n_momes = 32
    mome_requests = [
        MoMERequest(str(i + 1), i + 1, mome_adapter_files) for i in range(n_momes)
    ]

    worker.model_runner.set_active_momes(mome_requests, MoMEMapping([], []))
    assert worker.list_momes() == {
        mome_request.mome_int_id
        for mome_request in mome_requests
    }

    for i in range(32):
        random.seed(i)
        iter_mome_requests = random.choices(mome_requests,
                                            k=random.randint(1, n_momes))
        random.shuffle(iter_mome_requests)
        iter_mome_requests = iter_mome_requests[:-random.randint(0, n_momes)]
        worker.model_runner.set_active_momes(iter_mome_requests,
                                             MoMEMapping([], []))
        assert worker.list_momes().issuperset(
            {mome_request.mome_int_id
             for mome_request in iter_mome_requests})