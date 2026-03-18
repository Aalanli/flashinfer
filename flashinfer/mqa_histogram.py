"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.mqa_histogram import gen_mqa_histogram_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_mqa_histogram_module():
    module = gen_mqa_histogram_module().build_and_load()

    @register_custom_op("flashinfer::mqa_topk_indexer", mutates_args=("histogram", "logits", "indices"))
    def _mqa_topk_indexer(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        pdl_enabled: bool,
        sm_multiple: int,
    ) -> None:
        module.mqa_topk_indexer(
            q, k_cache, weights, seq_lens, block_table,
            histogram, sm_map, logits, indices, pdl_enabled, sm_multiple,
        )

    @register_fake_op("flashinfer::mqa_topk_indexer")
    def _fake_mqa_topk_indexer(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        pdl_enabled: bool,
        sm_multiple: int,
    ) -> None:
        pass

    @register_custom_op("flashinfer::get_mqa_metadata", mutates_args=())
    def _get_mqa_metadata(
        seq_lens: torch.Tensor,
        num_physical_sms: int,
    ) -> torch.Tensor:
        return module.get_mqa_metadata(seq_lens, num_physical_sms)

    @register_fake_op("flashinfer::get_mqa_metadata")
    def _fake_get_mqa_metadata(
        seq_lens: torch.Tensor,
        num_physical_sms: int,
    ) -> torch.Tensor:
        batch_size = seq_lens.size(0)
        num_logical_sms = (batch_size + num_physical_sms - 1) // num_physical_sms * num_physical_sms
        return torch.empty(num_logical_sms, 4, dtype=torch.int32, device=seq_lens.device)

    @register_custom_op("flashinfer::fast_topk_clusters_fused", mutates_args=("indices",))
    def _fast_topk_clusters_fused(
        logits: torch.Tensor,
        histogram: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_fused(logits, histogram, indices, seq_lens, pdl_enabled)

    @register_fake_op("flashinfer::fast_topk_clusters_fused")
    def _fake_fast_topk_clusters_fused(
        logits: torch.Tensor,
        histogram: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op("flashinfer::fast_topk_clusters", mutates_args=("indices",))
    def _fast_topk_clusters(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_cached: int,
        num_clusters: int,
    ) -> None:
        module.fast_topk_clusters(logits, indices, seq_lens, num_cached, num_clusters)

    @register_fake_op("flashinfer::fast_topk_clusters")
    def _fake_fast_topk_clusters(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_cached: int,
        num_clusters: int,
    ) -> None:
        pass

    @register_custom_op("flashinfer::mqa_logits", mutates_args=("logits",))
    def _mqa_logits(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        logits: torch.Tensor,
        sm_map: torch.Tensor,
        sm_multiple: int,
    ) -> None:
        module.mqa_logits(q, k_cache, weights, seq_lens, block_table, logits, sm_map, sm_multiple)

    @register_fake_op("flashinfer::mqa_logits")
    def _fake_mqa_logits(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        logits: torch.Tensor,
        sm_map: torch.Tensor,
        sm_multiple: int,
    ) -> None:
        pass

    @register_custom_op("flashinfer::mqa_logits_fused", mutates_args=("logits", "histogram"))
    def _mqa_logits_fused(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        logits: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        sm_multiple: int,
        pdl_enabled: bool,
    ) -> None:
        module.mqa_logits_fused(
            q, k_cache, weights, seq_lens, block_table,
            logits, histogram, sm_map, sm_multiple, pdl_enabled,
        )

    @register_fake_op("flashinfer::mqa_logits_fused")
    def _fake_mqa_logits_fused(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        logits: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        sm_multiple: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    return SimpleNamespace(
        mqa_topk_indexer=_mqa_topk_indexer,
        get_mqa_metadata=_get_mqa_metadata,
        fast_topk_clusters_fused=_fast_topk_clusters_fused,
        fast_topk_clusters=_fast_topk_clusters,
        mqa_logits=_mqa_logits,
        mqa_logits_fused=_mqa_logits_fused,
    )


@flashinfer_api
def get_mqa_metadata(seq_lens: torch.Tensor, num_sms: Optional[int] = None) -> torch.Tensor:
    """Compute SM mapping metadata for MQA load balancing.

    Args:
        seq_lens: [batch] int32 CUDA tensor of sequence lengths.
        num_sms: Number of physical SMs. If None, auto-detected from device.

    Returns:
        sm_map: [num_logical_sms, 4] int32 CUDA tensor.
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(seq_lens.device).multi_processor_count
    return get_mqa_histogram_module().get_mqa_metadata(seq_lens, num_sms)


@flashinfer_api
def mqa_topk_indexer_non_fused(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sm_map: Optional[torch.Tensor] = None,
    max_model_len: int = 163840,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Non-fused MQA top-K indexer: separate logit computation and top-K selection.

    Args:
        q:           [batch, 64, 128] fp8/uint8 CUDA tensor.
        k_cache:     [num_pages, 64, 1, 132] fp8/uint8 CUDA tensor.
        weights:     [batch, 64] float32 CUDA tensor.
        seq_lens:    [batch] int32 CUDA tensor.
        block_table: [batch, max_num_pages] int32 CUDA tensor.
        sm_map:      Optional [num_sms, 4] int32 tensor from get_mqa_metadata().
                     Auto-computed if None.
        max_model_len: Maximum sequence length to allocate logits buffer for.

    Returns:
        (indices, logits): indices [batch, 2048] int32, logits [batch, max_model_len] float32.
    """
    batch_size = q.shape[0]
    logits = torch.empty(batch_size, max_model_len, device=q.device, dtype=torch.float32)
    indices = torch.empty(batch_size, 2048, device=q.device, dtype=torch.int32)
    if sm_map is None:
        sm_map = get_mqa_metadata(seq_lens)

    ops = get_mqa_histogram_module()
    ops.mqa_logits(q, k_cache, weights, seq_lens, block_table, logits, sm_map, 1)
    ops.fast_topk_clusters(logits, indices, seq_lens, 4096, 8)

    return indices, logits


@flashinfer_api
def mqa_topk_indexer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sm_map: Optional[torch.Tensor] = None,
    max_model_len: int = 163840,
    pdl_enabled: bool = False,
    sm_multiple: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MQA top-K indexer: logit computation + histogram + top-K in one pass.

    Args:
        q:           [batch, 64, 128] fp8/uint8 CUDA tensor.
        k_cache:     [num_pages, 64, 1, 132] fp8/uint8 CUDA tensor.
        weights:     [batch, 64] float32 CUDA tensor.
        seq_lens:    [batch] int32 CUDA tensor.
        block_table: [batch, max_num_pages] int32 CUDA tensor.
        sm_map:      Optional [num_sms, 4] int32 tensor from get_mqa_metadata().
                     Auto-computed if None.
        max_model_len: Maximum sequence length to allocate logits buffer for.
        pdl_enabled: Enable Programmatic Launch Dependent (PDL) grid synchronization.
        sm_multiple: SM multiplier for load distribution.

    Returns:
        (indices, logits): indices [batch, 2048] int32, logits [batch, max_model_len] float32.
    """
    batch_size = q.shape[0]
    histogram = torch.zeros(batch_size, 256, device=q.device, dtype=torch.int32)
    logits = torch.empty(batch_size, max_model_len, device=q.device, dtype=torch.float32)
    indices = torch.empty(batch_size, 2048, device=q.device, dtype=torch.int32)
    if sm_map is None:
        sm_map = get_mqa_metadata(seq_lens)

    get_mqa_histogram_module().mqa_topk_indexer(
        q,
        k_cache,
        weights,
        seq_lens,
        block_table,
        histogram,
        sm_map,
        logits,
        indices,
        pdl_enabled,
        sm_multiple,
    )
    return indices, logits
