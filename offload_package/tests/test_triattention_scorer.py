"""Unit tests for the TriAttention token scorer.

The golden-parity tests validate the K_rot-direct scoring formula against
the methods-path reference (per-key positions + amp/phi reconstruction) that
TriAttention's HF implementation uses.  The two formulations are
mathematically identical because the key's position is baked into its RoPE
rotation; any divergence here means the port is wrong.
"""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring.token_base import TokenScorer, TokenRef
from offload_package.scoring.triattention import (
    TriAttentionTokenScorer,
    build_geometric_offsets,
    compute_rope_frequencies,
    load_triattention_stats,
)


# ---------------------------------------------------------------------------
# Reference implementation (methods path: unrotated keys + positions)
# ---------------------------------------------------------------------------


def _rotate(k_unrot: torch.Tensor, positions: torch.Tensor, omega: torch.Tensor, style: str) -> torch.Tensor:
    """Apply RoPE to unrotated keys at the given positions."""
    k = k_unrot.double()
    freq_count = k.shape[-1] // 2
    phase = positions.double().view(-1, 1, 1) * omega.double().view(1, 1, -1)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    if style == "interleaved":
        even, odd = k[..., 0::2], k[..., 1::2]
        out = torch.empty_like(k)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out
    real, imag = k[..., :freq_count], k[..., freq_count:]
    return torch.cat([real * cos - imag * sin, real * sin + imag * cos], dim=-1)


def _reference_score(
    k_unrot: torch.Tensor,
    positions: torch.Tensor,
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: int,
    *,
    style: str = "half",
    aggregation: str = "mean",
    disable_trig: bool = False,
    disable_mlr: bool = False,
) -> torch.Tensor:
    """R-KV-style reference: amp/phi from unrotated keys + key positions."""
    k = k_unrot.double()
    freq_count = k.shape[-1] // 2
    if style == "interleaved":
        k_complex = torch.complex(k[..., 0::2], k[..., 1::2])
    else:
        k_complex = torch.complex(k[..., :freq_count], k[..., freq_count:])
    q_mean_complex = torch.complex(
        q_mean_real.double(), q_mean_imag.double()
    )  # [H, F]
    fsq = freq_scale_sq.double()  # [H, F]

    scores_per_head = []
    for head in range(q_mean_complex.shape[0]):
        q_mean = q_mean_complex[head]  # [F]
        q_abs = q_mean.abs()
        k_abs = k_complex[:, head, :].abs()  # [n_tok, F]
        relative = q_mean.unsqueeze(0) * k_complex[:, head, :].conj()
        phi = torch.atan2(relative.imag, relative.real)
        amp = q_abs.unsqueeze(0) * k_abs
        if disable_mlr:
            extra = q_abs_mean[head].double().unsqueeze(0) * k_abs
        else:
            extra = (q_abs_mean[head].double() - q_abs).unsqueeze(0) * k_abs
        base_delta = float(round_start) - positions.double()
        delta_grid = base_delta.unsqueeze(1) + offsets.double().unsqueeze(0)
        phase = (
            delta_grid.unsqueeze(2) * omega.double().view(1, 1, -1)
            + phi.unsqueeze(1)
        )
        additive = (extra * fsq[head].view(1, -1)).sum(dim=1, keepdim=True)
        if disable_trig:
            combined = additive.expand(-1, delta_grid.shape[1])
        else:
            base_scores = (
                amp.unsqueeze(1) * fsq[head].view(1, 1, -1) * torch.cos(phase)
            ).sum(dim=2)
            combined = base_scores + additive
        if aggregation == "mean":
            scores_per_head.append(combined.mean(dim=1))
        else:
            scores_per_head.append(combined.max(dim=1).values)
    return torch.stack(scores_per_head, dim=0)  # [H, n_tok]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_TOK = 96
HEAD_DIM = 64
FREQ = HEAD_DIM // 2
N_HEADS = 2
ROPE_THETA = 10000.0


def _make_q_stats(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    q_mean_real = torch.randn(N_HEADS, FREQ, generator=g) * 0.5
    q_mean_imag = torch.randn(N_HEADS, FREQ, generator=g) * 0.5
    q_abs = (
        (q_mean_real**2 + q_mean_imag**2).sqrt()
        + torch.rand(N_HEADS, FREQ, generator=g) * 0.2
    )
    fsq = 1.0 + torch.rand(N_HEADS, FREQ, generator=g)
    return {
        "q_mean_real": q_mean_real,
        "q_mean_imag": q_mean_imag,
        "q_abs_mean": q_abs,
        "freq_scale_sq": fsq,
    }


def _write_tri_stats(tmp_path, q_stats: dict, *, rope_style: str = "half",
                      layers: int = 1, head_dim: int = HEAD_DIM,
                      extra_metadata: dict | None = None) -> str:
    """Write a stats file in the TriAttention runtime format."""
    payload = {
        "metadata": {
            "num_attention_heads": N_HEADS,
            "num_kv_heads": N_HEADS,
            "head_dim": head_dim,
            "num_layers": layers,
            "rope_theta": ROPE_THETA,
            "rope_style": rope_style,
            **(extra_metadata or {}),
        },
        "layer_stats": {},
    }
    for layer in range(layers):
        q_mean_complex = torch.stack(
            [q_stats["q_mean_real"], q_stats["q_mean_imag"]], dim=-1
        )
        payload["layer_stats"][layer] = {
            "q_mean_complex": q_mean_complex,
            "q_abs_mean": q_stats["q_abs_mean"],
            "freq_scale_sq": q_stats["freq_scale_sq"],
        }
    path = tmp_path / "tri_stats.pt"
    torch.save(payload, path)
    return str(path)


def _make_inputs(seed: int = 1, style: str = "half"):
    g = torch.Generator().manual_seed(seed)
    k_unrot = torch.randn(N_TOK, N_HEADS, HEAD_DIM, generator=g) * 0.5
    positions = torch.arange(1000, 1000 + N_TOK)
    omega = compute_rope_frequencies(HEAD_DIM, ROPE_THETA, torch.device("cpu"))
    k_rot = _rotate(k_unrot, positions, omega, style).float()
    return k_unrot, positions, omega, k_rot


# ---------------------------------------------------------------------------
# Golden parity tests
# ---------------------------------------------------------------------------


class TestGoldenParity:

    def _run_parity(self, tmp_path, torch_device, *, style="half",
                    aggregation="mean", disable_trig=False, disable_mlr=False,
                    seed=0):
        q = _make_q_stats(seed)
        path = _write_tri_stats(tmp_path, q, rope_style=style)
        k_unrot, positions, omega, k_rot = _make_inputs(style=style)
        round_start = int(positions[-1]) + 7

        scorer = TriAttentionTokenScorer(
            path,
            torch_device,
            score_aggregation=aggregation,
            disable_trig=disable_trig,
            disable_mlr=disable_mlr,
            offsets_max_length=4096,
        )
        offsets = build_geometric_offsets(4096, torch.device("cpu"))
        ours = scorer.score({0: k_rot.to(torch_device)}, round_start).cpu()

        ref_heads = _reference_score(
            k_unrot,
            positions,
            q["q_mean_real"],
            q["q_mean_imag"],
            q["q_abs_mean"],
            q["freq_scale_sq"],
            omega,
            offsets,
            round_start,
            style=style,
            aggregation=aggregation,
            disable_trig=disable_trig,
            disable_mlr=disable_mlr,
        )
        ref = ref_heads.mean(dim=0).float()  # head_aggregation="mean"
        torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)
        return ours, ref

    def test_matches_reference_half(self, torch_device, tmp_path):
        ours, ref = self._run_parity(tmp_path, torch_device)
        assert ours.shape == (N_TOK,)
        assert not torch.allclose(ours, torch.zeros_like(ours))

    def test_matches_reference_interleaved(self, torch_device, tmp_path):
        self._run_parity(tmp_path, torch_device, style="interleaved", seed=3)

    def test_matches_reference_max_aggregation(self, torch_device, tmp_path):
        self._run_parity(tmp_path, torch_device, aggregation="max", seed=4)

    def test_matches_reference_disable_trig(self, torch_device, tmp_path):
        ours, ref = self._run_parity(tmp_path, torch_device, disable_trig=True)
        # MLR-only scores must still vary across tokens.
        assert (ref.std() > 0).item()

    def test_matches_reference_disable_mlr(self, torch_device, tmp_path):
        self._run_parity(tmp_path, torch_device, disable_mlr=True, seed=5)

    def test_matches_reference_combined(self, torch_device, tmp_path):
        self._run_parity(
            tmp_path, torch_device, disable_trig=True, disable_mlr=True, seed=6
        )

    def test_multi_layer_head_mean(self, torch_device, tmp_path):
        """Multi-layer stats must aggregate across all (layer, head) pairs."""
        q0 = _make_q_stats(0)
        q1 = _make_q_stats(7)
        # Two-layer file: layer 0 uses q0, layer 1 uses q1.
        payload = {
            "metadata": {
                "num_attention_heads": N_HEADS,
                "num_kv_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": 2,
                "rope_theta": ROPE_THETA,
                "rope_style": "half",
            },
            "layer_stats": {
                0: {
                    "q_mean_complex": torch.stack(
                        [q0["q_mean_real"], q0["q_mean_imag"]], dim=-1
                    ),
                    "q_abs_mean": q0["q_abs_mean"],
                    "freq_scale_sq": q0["freq_scale_sq"],
                },
                1: {
                    "q_mean_complex": torch.stack(
                        [q1["q_mean_real"], q1["q_mean_imag"]], dim=-1
                    ),
                    "q_abs_mean": q1["q_abs_mean"],
                    "freq_scale_sq": q1["freq_scale_sq"],
                },
            },
        }
        path = tmp_path / "multi.pt"
        torch.save(payload, path)

        k_unrot, positions, omega, k_rot = _make_inputs()
        round_start = int(positions[-1]) + 7
        scorer = TriAttentionTokenScorer(
            str(path), torch_device, offsets_max_length=4096
        )
        ours = scorer.score({0: k_rot, 1: k_rot}, round_start).cpu()

        offsets = build_geometric_offsets(4096, torch.device("cpu"))
        ref0 = _reference_score(
            k_unrot, positions, q0["q_mean_real"], q0["q_mean_imag"],
            q0["q_abs_mean"], q0["freq_scale_sq"], omega, offsets, round_start,
        )
        ref1 = _reference_score(
            k_unrot, positions, q1["q_mean_real"], q1["q_mean_imag"],
            q1["q_abs_mean"], q1["freq_scale_sq"], omega, offsets, round_start,
        )
        ref = torch.cat([ref0, ref1], dim=0).mean(dim=0).float()
        torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)

    def test_max_head_aggregation(self, torch_device, tmp_path):
        q = _make_q_stats(0)
        path = _write_tri_stats(tmp_path, q)
        k_unrot, positions, omega, k_rot = _make_inputs()
        round_start = int(positions[-1]) + 7
        offsets = build_geometric_offsets(4096, torch.device("cpu"))

        scorer = TriAttentionTokenScorer(
            path, torch_device, head_aggregation="max", offsets_max_length=4096
        )
        ours = scorer.score({0: k_rot}, round_start).cpu()
        ref = _reference_score(
            k_unrot, positions, q["q_mean_real"], q["q_mean_imag"],
            q["q_abs_mean"], q["freq_scale_sq"], omega, offsets, round_start,
        ).max(dim=0).values.float()
        torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoadStats:

    def test_tri_format_roundtrip(self, torch_device, tmp_path):
        q = _make_q_stats()
        path = _write_tri_stats(tmp_path, q)
        stats = load_triattention_stats(path, torch_device)
        assert stats.rope_style == "half"
        assert stats.head_dim == HEAD_DIM
        assert sorted(stats.layers) == [0]
        layer = stats.layers[0]
        assert layer.num_kv_heads == N_HEADS
        assert layer.freq_count == FREQ
        torch.testing.assert_close(
            layer.q_mean_real, q["q_mean_real"].to(torch_device)
        )
        torch.testing.assert_close(
            layer.freq_scale_sq, q["freq_scale_sq"].to(torch_device)
        )

    def test_tri_format_derives_q_abs_when_missing(self, torch_device, tmp_path):
        q = _make_q_stats()
        payload = {
            "metadata": {
                "num_attention_heads": N_HEADS,
                "num_kv_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": 1,
                "rope_theta": ROPE_THETA,
                "rope_style": "half",
            },
            "layer_stats": {
                0: {
                    "q_mean_complex": torch.stack(
                        [q["q_mean_real"], q["q_mean_imag"]], dim=-1
                    ),
                    "freq_scale_sq": q["freq_scale_sq"],
                }
            },
        }
        path = tmp_path / "no_abs.pt"
        torch.save(payload, path)
        stats = load_triattention_stats(str(path), torch_device)
        derived = stats.layers[0].q_abs_mean
        expected = (
            (q["q_mean_real"] ** 2 + q["q_mean_imag"] ** 2 + 1e-8).sqrt()
        )
        torch.testing.assert_close(derived, expected.to(torch_device))

    def test_rkv_format_gqa_averaging(self, torch_device, tmp_path):
        """8 sampled attn heads -> 2 KV heads via group averaging."""
        n_attn, n_kv = 8, 2
        g = torch.Generator().manual_seed(11)
        stats_entries = {}
        for h in range(n_attn):
            stats_entries[f"layer00_head{h:02d}"] = {
                "q_mean_real": torch.randn(FREQ, generator=g),
                "q_mean_imag": torch.randn(FREQ, generator=g),
                "q_abs_mean": torch.rand(FREQ, generator=g) + 0.5,
            }
        payload = {
            "metadata": {
                "num_attention_heads": n_attn,
                "num_kv_heads": n_kv,
                "head_dim": HEAD_DIM,
                "rope_theta": ROPE_THETA,
                "rope_style": "half",
            },
            "stats": stats_entries,
        }
        path = tmp_path / "rkv.pt"
        torch.save(payload, path)
        loaded = load_triattention_stats(str(path), torch_device, num_kv_heads=n_kv)
        layer = loaded.layers[0]
        assert layer.num_kv_heads == n_kv
        expected_kv0_abs = torch.stack(
            [stats_entries[f"layer00_head{h:02d}"]["q_abs_mean"] for h in range(4)]
        ).mean(dim=0)
        torch.testing.assert_close(
            layer.q_abs_mean[0].cpu(), expected_kv0_abs, rtol=1e-5, atol=1e-6
        )
        expected_kv1_abs = torch.stack(
            [stats_entries[f"layer00_head{h:02d}"]["q_abs_mean"] for h in range(4, 8)]
        ).mean(dim=0)
        torch.testing.assert_close(
            layer.q_abs_mean[1].cpu(), expected_kv1_abs, rtol=1e-5, atol=1e-6
        )

    def test_rkv_format_sampled_subset(self, torch_device, tmp_path):
        """Sparse head sampling must still fill every KV-head bucket."""
        entries = {
            "layer00_head01": {
                "q_mean_real": torch.ones(FREQ),
                "q_mean_imag": torch.zeros(FREQ),
                "q_abs_mean": torch.full((FREQ,), 2.0),
            },
            "layer00_head05": {
                "q_mean_real": torch.zeros(FREQ),
                "q_mean_imag": torch.ones(FREQ),
                "q_abs_mean": torch.full((FREQ,), 4.0),
            },
        }
        payload = {
            "metadata": {
                "num_attention_heads": 8,
                "num_kv_heads": 2,
                "head_dim": HEAD_DIM,
                "rope_theta": ROPE_THETA,
            },
            "stats": entries,
        }
        path = tmp_path / "rkv_subset.pt"
        torch.save(payload, path)
        loaded = load_triattention_stats(str(path), torch_device, num_kv_heads=2)
        layer = loaded.layers[0]
        assert layer.num_kv_heads == 2
        # gqa_ratio=4: head 1 -> kv 0, head 5 -> kv 1.
        assert layer.q_abs_mean[0].max().item() == pytest.approx(2.0)
        assert layer.q_abs_mean[1].max().item() == pytest.approx(4.0)

    def test_metadata_inv_freq_respected(self, torch_device, tmp_path):
        q = _make_q_stats()
        custom = torch.linspace(1.0, 0.01, FREQ)
        path = _write_tri_stats(
            tmp_path, q, extra_metadata={"inv_freq": custom}
        )
        loaded = load_triattention_stats(path, torch_device)
        torch.testing.assert_close(
            loaded.omega.cpu(), custom, rtol=1e-5, atol=1e-6
        )

    def test_missing_file_raises(self, torch_device, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_triattention_stats(tmp_path / "nope.pt", torch_device)

    def test_garbage_payload_raises(self, torch_device, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"metadata": {}}, path)
        with pytest.raises(ValueError):
            load_triattention_stats(str(path), torch_device)

    def test_tri_format_missing_metadata_raises(self, torch_device, tmp_path):
        q = _make_q_stats()
        payload = {
            "metadata": {"head_dim": HEAD_DIM},
            "layer_stats": {
                0: {
                    "q_mean_complex": torch.stack(
                        [q["q_mean_real"], q["q_mean_imag"]], dim=-1
                    )
                }
            },
        }
        path = tmp_path / "missing_meta.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="missing required keys"):
            load_triattention_stats(str(path), torch_device)


# ---------------------------------------------------------------------------
# Scorer interface tests
# ---------------------------------------------------------------------------


class TestScorerInterface:

    def _scorer(self, torch_device, tmp_path, **kwargs):
        q = _make_q_stats()
        path = _write_tri_stats(tmp_path, q)
        return TriAttentionTokenScorer(path, torch_device, **kwargs)

    def test_empty_gather(self, torch_device, tmp_path):
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        out = scorer.score({}, 100)
        assert out.shape == (0,)
        k = torch.zeros(0, N_HEADS, HEAD_DIM, device=torch_device)
        assert scorer.score({0: k}, 100).shape == (0,)

    def test_layers_missing_from_gather_are_skipped(
        self, torch_device, tmp_path
    ):
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        k = torch.randn(4, N_HEADS, HEAD_DIM, device=torch_device)
        # Stats only define layer 0; layer 7 in the gather is ignored.
        out = scorer.score({0: k, 7: k}, 50)
        assert out.shape == (4,)
        assert not torch.isnan(out).any()

    def test_head_count_mismatch_raises(self, torch_device, tmp_path):
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        bad = torch.randn(4, N_HEADS + 1, HEAD_DIM, device=torch_device)
        with pytest.raises(ValueError, match="kv heads"):
            scorer.score({0: bad}, 50)

    def test_head_dim_mismatch_raises(self, torch_device, tmp_path):
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        bad = torch.randn(4, N_HEADS, HEAD_DIM + 2, device=torch_device)
        with pytest.raises(ValueError, match="head_dim"):
            scorer.score({0: bad}, 50)

    def test_wrong_ndim_raises(self, torch_device, tmp_path):
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        bad = torch.randn(4, HEAD_DIM, device=torch_device)
        with pytest.raises(ValueError, match="num_kv_heads"):
            scorer.score({0: bad}, 50)

    def test_invalid_aggregation_raises(self, torch_device, tmp_path):
        q = _make_q_stats()
        path = _write_tri_stats(tmp_path, q)
        with pytest.raises(ValueError, match="head_aggregation"):
            TriAttentionTokenScorer(
                path, torch_device, head_aggregation="sum"
            )
        with pytest.raises(ValueError, match="score_aggregation"):
            TriAttentionTokenScorer(
                path, torch_device, score_aggregation="sum"
            )

    def test_scores_cpu_and_gpu_tensors(self, torch_device, tmp_path):
        if torch_device.type != "cuda":
            pytest.skip("requires CUDA to exercise cross-device scoring")
        scorer = self._scorer(torch_device, tmp_path, offsets_max_length=1024)
        k_gpu = torch.randn(8, N_HEADS, HEAD_DIM, device="cuda")
        k_cpu = k_gpu.cpu()
        round_start = 512
        s_gpu = scorer.score({0: k_gpu}, round_start)
        s_cpu = scorer.score({0: k_cpu}, round_start)
        assert s_cpu.device.type == "cpu"
        torch.testing.assert_close(s_cpu, s_gpu.cpu(), rtol=1e-4, atol=1e-4)

    def test_token_ref_identity(self):
        a, b = TokenRef("r", 0), TokenRef("r", 0)
        assert a == b
        assert hash(a) == hash(b)
        assert a != TokenRef("r", 1)

    def test_token_scorer_abc(self):
        with pytest.raises(TypeError):
            TokenScorer()  # type: ignore[abstract]
