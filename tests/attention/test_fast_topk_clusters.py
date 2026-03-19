"""
Tests for the fast_topk_clusters kernel.

Covers two modes:
  - without pre-histogram (histogram=None): kernel builds the first-pass histogram itself
  - with pre-histogram (histogram provided): kernel reuses a caller-supplied histogram

Requires SM100a (Blackwell) GPU.
"""

import pytest
import torch

from flashinfer.mqa_histogram import get_mqa_histogram_module
from flashinfer.utils import is_sm100a_supported

BATCH_SIZES = [1, 4, 8, 32, 64]
SEQ_LENS = [32] + [4096, 8192] + [8192 * 2 * i for i in range(2, 9)]


def _compute_hist(logits: torch.Tensor) -> torch.Tensor:
    """Compute the most-significant-byte histogram used by the fused top-K path."""
    B, L = logits.shape
    hist = torch.zeros(B, 256, dtype=torch.int32, device=logits.device)
    bits = logits.view(torch.int32)
    bits = torch.where((bits & 0x80000000) != 0, ~bits, (bits | 0x80000000))
    bins = (bits.view(torch.uint8).view(-1, 4)[:, 3]).to(torch.int32).view(B, L)
    ones = torch.ones(L, dtype=torch.int32, device=logits.device)
    for i in range(B):
        hist[i].index_add_(0, bins[i], ones)
    return hist


def _assert_topk_indices(logits: torch.Tensor, topk_inds: torch.Tensor) -> None:
    """Assert that topk_inds contains the correct top-2048 indices."""
    B, L = logits.shape
    if L <= 2048:
        return
    for i in range(B):
        topk_ref, _ = torch.topk(logits[i], 2048)
        topk_inds_i = topk_inds[i][topk_inds[i] >= 0]
        assert torch.unique(topk_inds_i).shape[0] == topk_inds_i.shape[0], (
            f"repeated indices at batch index {i}"
        )
        topk_actual = logits[i][topk_inds_i]
        topk_ref = torch.sort(topk_ref)[0]
        topk_actual = torch.sort(topk_actual)[0]
        diff = (topk_ref - topk_actual).abs().max()
        assert diff < 0.1, f"topk diff too large at batch idx {i}: {diff}"


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_clusters", [1, 2, 4, 8])
@pytest.mark.parametrize("num_cached", [4096, 8192])
def test_fast_topk_clusters_with_histogram(
    batch_size, seq_len, num_clusters, num_cached
):
    """fast_topk_clusters with a pre-supplied histogram (PRE_HISTOGRAM=true path)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")
    if seq_len >= 16384 and num_clusters < 4:
        pytest.skip("not enough cached objects for this (seq_len, num_clusters) combo")

    torch.manual_seed(1)
    logits = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)
    histogram = _compute_hist(logits)
    indices = torch.empty(batch_size, 2048, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)

    get_mqa_histogram_module().fast_topk_clusters(
        logits, indices, seq_lens, histogram, num_cached, num_clusters, False
    )

    assert indices.shape == (batch_size, 2048)
    valid = indices[indices >= 0]
    assert (valid < seq_len).all(), "indices out of range"
    _assert_topk_indices(logits, indices)


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_clusters", [1, 2, 4, 8])
@pytest.mark.parametrize("num_cached", [4096, 8192])
def test_fast_topk_clusters_no_histogram(batch_size, seq_len, num_clusters, num_cached):
    """fast_topk_clusters without pre-histogram (PRE_HISTOGRAM=false path)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")
    if seq_len >= 16384 and num_clusters < 4:
        pytest.skip("not enough cached objects for this (seq_len, num_clusters) combo")

    torch.manual_seed(1)
    logits = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)
    indices = torch.empty(batch_size, 2048, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)

    get_mqa_histogram_module().fast_topk_clusters(
        logits, indices, seq_lens, None, num_cached, num_clusters, False
    )

    assert indices.shape == (batch_size, 2048)
    valid = indices[indices >= 0]
    assert (valid < seq_len).all(), "indices out of range"
    _assert_topk_indices(logits, indices)
