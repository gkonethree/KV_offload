import math

import pytest
import torch

from original import BatchDecodeWithPagedKVCacheWrapper as OriginalWrapper


def _get_flashinfer_wrapper_cls():
    flashinfer = pytest.importorskip("flashinfer")
    wrapper_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if wrapper_cls is not None:
        return wrapper_cls
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper


def _make_synthetic_decode_inputs(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    pages_per_req = math.ceil(context_len / page_size)
    total_pages = batch_size * pages_per_req
    last_page_len_value = context_len % page_size or page_size

    kv_indptr = torch.arange(
        0, total_pages + 1, pages_per_req, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), last_page_len_value, dtype=torch.int32, device=device
    )
    kv_cache = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashInfer.")
def test_original_matches_flashinfer_decode_run():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    batch_size = 4
    context_len = 256
    page_size = 16
    num_kv_heads = 4
    num_q_heads = 16
    head_dim = 64

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_synthetic_decode_inputs(
        batch_size=batch_size,
        context_len=context_len,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    workspace_flashinfer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    workspace_original = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    FlashinferWrapper = _get_flashinfer_wrapper_cls()
    flash_wrapper = FlashinferWrapper(workspace_flashinfer, "NHD")
    original_wrapper = OriginalWrapper(workspace_original, "NHD")

    flash_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    original_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    flash_out, flash_lse = flash_wrapper.run(q, kv_cache, return_lse=True)
    orig_out, orig_lse = original_wrapper.run(q, kv_cache, return_lse=True)

    assert flash_out.shape == orig_out.shape
    assert flash_lse.shape == orig_lse.shape
    assert torch.allclose(orig_out, flash_out, rtol=5e-2, atol=5e-2)
    assert torch.allclose(orig_lse, flash_lse, rtol=5e-2, atol=5e-2)
