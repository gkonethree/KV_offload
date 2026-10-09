import torch

from offload_package.controller import KVCompressionController
from offload_package.scoring.base import BlockKVStats, BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.selection.adapter import patch_paged_kv_indices_with_staging
from offload_package.staging.residency import Residency, ResidencyTable


def test_decay_semantics():
    scorer = PagedEvictionScorer(decay=1.0)
    refs = [BlockRef("r", 0)]
    scorer.update(BlockKVStats(torch.tensor([[1.0]]), torch.tensor([[1.0]]), refs))
    scorer.update(BlockKVStats(torch.tensor([[4.0]]), torch.tensor([[1.0]]), refs))
    assert torch.isclose(scorer.scores(refs), torch.tensor([4.0])).item()


def test_per_request_interval():
    c = KVCompressionController(W=4, page_size=2)
    c.on_prefill_complete("a")
    c.on_prefill_complete("b")
    assert not c.on_decode("a")
    assert not c.on_decode("a")
    assert not c.on_decode("a")
    assert c.on_decode("a")
    assert c.count("b") == 0


def test_flat_paged_indices_patch_uses_indptr():
    rt = ResidencyTable()
    ref = BlockRef("req1", 1)
    rt.mark_gpu(ref, 7, complete=True)
    rt.mark_complete(ref)
    # Pretend the block is already on CPU, then staging slot 3.
    rt.mark_cpu(ref, 0)
    rt.mark_staging(ref, 3)

    indices = torch.tensor([10, 11, 20, 21], dtype=torch.int32)
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    sparse_idx = torch.tensor([[[2], [2]], [[0], [0]]], dtype=torch.int64)
    sparse_len = torch.ones((2, 2, 1), dtype=torch.int32)
    patched = patch_paged_kv_indices_with_staging(
        indices,
        indptr,
        ["req1", "req2"],
        sparse_idx,
        sparse_len,
        2,
        100,
        rt,
        {ref: 3},
    )
    assert patched.tolist() == [10, 103, 20, 21]


def test_patch_remediation_only_without_sparse_pattern():
    """sparse_idx=None must still run the residency remediation: STAGING ->
    its slot, CPU -> null page (under true freeing), GPU -> untouched. This
    is the path taken when the lag pattern belongs to a previous batch
    composition; skipping remediation would leave freed page ids in the
    kernel's index buffer."""
    rt = ResidencyTable()
    staged = BlockRef("req1", 0)
    cold = BlockRef("req1", 1)
    live = BlockRef("req1", 2)
    for ref in (staged, cold, live):
        rt.mark_gpu(ref, 7 + ref.block_idx, complete=True)
    rt.mark_cpu(staged, 0)
    rt.mark_staging(staged, 3)
    rt.mark_cpu(cold, 1)

    indices = torch.tensor([10, 11, 12], dtype=torch.int32)
    indptr = torch.tensor([0, 3], dtype=torch.int32)
    patched = patch_paged_kv_indices_with_staging(
        indices,
        indptr,
        ["req1"],
        None,   # no cached sparse pattern
        None,
        1,      # page_size
        100,    # staging_base
        rt,
        {},     # no newly staged slots
        null_block_id=0,
    )
    assert patched.tolist() == [103, 0, 12]


def test_patch_remediation_skips_mismatched_row():
    """A row whose length no longer matches the tracked block count must be
    skipped entirely rather than written with wrong positions."""
    rt = ResidencyTable()
    staged = BlockRef("req1", 0)
    cold = BlockRef("req1", 1)
    live = BlockRef("req1", 2)
    for ref in (staged, cold, live):
        rt.mark_gpu(ref, 7 + ref.block_idx, complete=True)
    rt.mark_cpu(staged, 0)
    rt.mark_staging(staged, 3)
    rt.mark_cpu(cold, 1)

    indices = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32)
    indptr = torch.tensor([0, 3, 5], dtype=torch.int32)
    patched = patch_paged_kv_indices_with_staging(
        indices,
        indptr,
        ["req1", "req2"],   # req2 has no tracked blocks
        None,
        None,
        1,
        100,
        rt,
        {},
        null_block_id=0,
    )
    # req1's row remediated; req2's row (2 pages, 0 tracked) untouched.
    assert patched.tolist() == [103, 0, 12, 20, 21]
