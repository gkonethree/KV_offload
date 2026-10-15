"""TriAttention token scorer.

Ports the frequency-domain scoring of TriAttention
(https://github.com/WeianMao/triattention) to the offload package's
``TokenScorer`` interface.  Two properties make this scorer a good fit for
token-level offloading:

1. It never reads attention scores (which FlashAttention-style kernels do
   not expose).  Importance comes from the geometry of the model's learned
   Q/K vectors, summarized in per-(layer, head) calibration statistics.
2. The score operates directly on the *post-RoPE* keys stored in the KV
   cache.  Because the position of each key is baked into its rotation, the
   query-side phase is the only position-dependent input, so scoring stays
   correct no matter where a token physically lives (GPU page, CPU offload
   store, staging area) or how often it has been compacted.

The dense scoring formula follows TriAttention's vLLM-core reference
(``compute_scores_pytorch``):

    score = agg_over_offsets( sum_f freq_scale^2 * (A*cos(t*w) - B*sin(t*w)) )
            + sum_f freq_scale^2 * extra_coef * |K|

where ``A + iB = Q_mean * conj(K_rot)``, ``t = round_start + offset`` and
``extra_coef`` is the magnitude-LR term from the calibration stats.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .token_base import TokenScorer

_REQUIRED_TRI_METADATA = (
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "num_layers",
)


@dataclass
class LayerHeadStats:
    """Per-layer scoring statistics, already GQA-mapped to KV heads."""

    q_mean_real: torch.Tensor     # [num_kv_heads, freq_count]
    q_mean_imag: torch.Tensor     # [num_kv_heads, freq_count]
    q_abs_mean: torch.Tensor      # [num_kv_heads, freq_count]
    freq_scale_sq: torch.Tensor   # [num_kv_heads, freq_count]

    @property
    def num_kv_heads(self) -> int:
        return int(self.q_mean_real.shape[0])

    @property
    def freq_count(self) -> int:
        return int(self.q_mean_real.shape[1])


@dataclass
class TriAttentionStats:
    """Calibration statistics for the TriAttention scorer."""

    head_dim: int
    rope_style: str  # "half" (Qwen/Llama) or "interleaved"
    omega: torch.Tensor  # [freq_count] shared rotation frequencies
    layers: dict[int, LayerHeadStats] = field(default_factory=dict)
    layer_omega: dict[int, torch.Tensor] = field(default_factory=dict)
    num_kv_heads: int | None = None


def compute_rope_frequencies(
    head_dim: int,
    rope_theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )


def build_geometric_offsets(
    max_length: int, device: torch.device
) -> torch.Tensor:
    """Reference query offsets [1, 2, 4, ..., <= max_length]."""
    if max_length < 1:
        raise ValueError(f"offsets_max_length must be >= 1, got {max_length}")
    offsets: list[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _fit_1d(value: torch.Tensor, target: int, *, fill: float = 0.0) -> torch.Tensor:
    value = value.flatten().to(dtype=torch.float32)
    if value.numel() == target:
        return value
    if value.numel() > target:
        return value[:target]
    out = torch.full((target,), fill, dtype=torch.float32, device=value.device)
    out[: value.numel()] = value
    return out


def _inv_freq_from_metadata(metadata: dict, max_freq_count: int, device: torch.device):
    """Resolve rotation frequencies from metadata, else rope_theta fallback."""
    inv_freq_raw = metadata.get("inv_freq")
    if isinstance(inv_freq_raw, torch.Tensor):
        inv_freq = inv_freq_raw.to(device=device, dtype=torch.float32)
    elif isinstance(inv_freq_raw, (list, tuple)):
        inv_freq = torch.tensor(inv_freq_raw, device=device, dtype=torch.float32)
    else:
        rope_theta = float(metadata.get("rope_theta", 10000.0))
        head_dim = int(metadata.get("head_dim", max_freq_count * 2))
        if head_dim <= 0:
            head_dim = max_freq_count * 2
        inv_freq = compute_rope_frequencies(head_dim, rope_theta, device)
    if inv_freq.numel() < 1:
        raise ValueError("inv_freq must be non-empty")
    return inv_freq[:max_freq_count].contiguous()


def _convert_rkv_stats(
    metadata: dict,
    rkv_stats: dict,
    device: torch.device,
    num_kv_heads: int | None,
) -> tuple[dict, dict[int, LayerHeadStats]]:
    """Convert R-KV per-(layer, attn-head) stats to per-layer KV-head stats.

    Sampled attention heads are bucketed onto KV heads via their GQA group
    (``attn_head // gqa_ratio``) and averaged within each bucket.  Stats
    files may sample only a subset of the model's attention heads, so
    buckets left empty fall back to the mean of all sampled heads.
    """
    layer_nums: set[int] = set()
    head_nums: set[int] = set()
    for key in rkv_stats:
        parts = key.split("_")
        if len(parts) == 2 and key.startswith("layer"):
            layer_nums.add(int(parts[0].removeprefix("layer")))
            head_nums.add(int(parts[1].removeprefix("head")))
    if not layer_nums:
        raise ValueError("R-KV stats contain no 'layerXX_headYY' entries")

    sampled_attn = len(head_nums)
    if num_kv_heads is None:
        num_kv_heads = int(
            metadata.get("num_kv_heads", metadata.get("num_key_value_heads", 0))
            or sampled_attn
        )
    if num_kv_heads <= 0:
        raise ValueError(f"invalid num_kv_heads: {num_kv_heads}")
    model_attn = int(metadata.get("num_attention_heads", 0)) or sampled_attn
    if model_attn >= num_kv_heads and model_attn % num_kv_heads == 0:
        gqa_ratio = model_attn // num_kv_heads
        kv_bucket = {h: h // gqa_ratio for h in head_nums}
    else:
        kv_bucket = {h: h % num_kv_heads for h in head_nums}

    def _entry_freq_count(layer_idx: int) -> int:
        for head_idx in sorted(head_nums):
            entry = rkv_stats.get(f"layer{layer_idx:02d}_head{head_idx:02d}")
            if not isinstance(entry, dict):
                continue
            for stat_name in ("q_abs_mean", "q_mean_real", "q_mean_imag"):
                value = entry.get(stat_name)
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    return int(value.numel())
        return max(1, int(metadata.get("head_dim", 128)) // 2)

    def _layer_entry(layer_idx: int, head_idx: int) -> dict:
        return rkv_stats.get(f"layer{layer_idx:02d}_head{head_idx:02d}", {})

    def _bucket_mean(stacked: torch.Tensor) -> torch.Tensor:
        """[sampled_heads, F] -> [num_kv_heads, F] with per-bucket means."""
        out = stacked.new_zeros(num_kv_heads, stacked.shape[1])
        counted = torch.zeros(num_kv_heads, dtype=torch.int64)
        for row, head_idx in enumerate(sorted(head_nums)):
            j = kv_bucket[head_idx]
            out[j] += stacked[row]
            counted[j] += 1
        empty = counted == 0
        if bool(empty.any()):
            out[empty] = stacked.mean(dim=0)
            counted[empty] = 1
        return out / counted.unsqueeze(1)

    layers: dict[int, LayerHeadStats] = {}
    per_layer_freq: dict[int, int] = {}
    for layer_idx in sorted(layer_nums):
        freq_count = _entry_freq_count(layer_idx)
        per_layer_freq[layer_idx] = freq_count
        have_mean = all(
            ("q_mean_real" in _layer_entry(layer_idx, h))
            and ("q_mean_imag" in _layer_entry(layer_idx, h))
            for h in sorted(head_nums)
        )
        if not have_mean:
            # Without per-head Q means the trigonometric term degenerates;
            # the MLR-only score is still a usable magnitude ranking.
            q_mean_real = torch.zeros(num_kv_heads, freq_count)
            q_mean_imag = torch.zeros(num_kv_heads, freq_count)
        else:
            group_real = torch.stack(
                [
                    _fit_1d(
                        _layer_entry(layer_idx, h)["q_mean_real"], freq_count
                    )
                    for h in sorted(head_nums)
                ],
                dim=0,
            )
            group_imag = torch.stack(
                [
                    _fit_1d(
                        _layer_entry(layer_idx, h)["q_mean_imag"], freq_count
                    )
                    for h in sorted(head_nums)
                ],
                dim=0,
            )
            q_mean_real = _bucket_mean(group_real)
            q_mean_imag = _bucket_mean(group_imag)

        q_abs_stack = torch.stack(
            [
                _fit_1d(
                    _layer_entry(layer_idx, h).get(
                        "q_abs_mean", torch.ones(freq_count)
                    ),
                    freq_count,
                    fill=1.0,
                )
                for h in sorted(head_nums)
            ],
            dim=0,
        )
        q_abs_mean = _bucket_mean(q_abs_stack)
        layers[layer_idx] = LayerHeadStats(
            q_mean_real=q_mean_real.to(device),
            q_mean_imag=q_mean_imag.to(device),
            q_abs_mean=q_abs_mean.to(device),
            freq_scale_sq=torch.ones(num_kv_heads, freq_count, device=device),
        )

    converted_meta = dict(metadata)
    converted_meta.setdefault("head_dim", max(per_layer_freq.values()) * 2)
    converted_meta["num_kv_heads"] = num_kv_heads
    converted_meta["gqa_ratio"] = gqa_ratio
    converted_meta["layer_freq_counts"] = per_layer_freq
    return converted_meta, layers


def load_triattention_stats(
    stats_path: str | Path,
    device: torch.device,
    num_kv_heads: int | None = None,
) -> TriAttentionStats:
    """Load calibration statistics from either supported file format.

    Supports the TriAttention runtime format (per-layer stacked KV-head
    tensors) and the R-KV format (per-(layer, attn-head) entries, converted
    with GQA averaging).
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"stats file not found: {stats_path}")

    payload = torch.load(stats_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("stats 'metadata' must be a dict")

    rkv_stats = payload.get("stats", {})
    is_rkv_format = bool(rkv_stats) and any(
        k.startswith("layer") and "_head" in k for k in rkv_stats
    )
    if is_rkv_format:
        metadata, layers = _convert_rkv_stats(
            metadata, rkv_stats, device, num_kv_heads
        )
    else:
        layer_stats = payload.get("layer_stats", {})
        if not layer_stats:
            raise ValueError("stats file contains neither 'stats' nor 'layer_stats'")
        missing = [k for k in _REQUIRED_TRI_METADATA if k not in metadata]
        if missing:
            raise ValueError(
                f"stats metadata missing required keys: {missing}; "
                f"found: {list(metadata)}"
            )
        layers = {}
        for layer_idx, layer_data in layer_stats.items():
            if not isinstance(layer_data, dict):
                continue
            q_mean = layer_data.get("q_mean_complex")
            if not isinstance(q_mean, torch.Tensor) or q_mean.ndim != 3:
                raise ValueError(
                    f"layer {layer_idx}: 'q_mean_complex' must be "
                    f"[num_kv_heads, freq_count, 2]"
                )
            h, freq_count = q_mean.shape[0], q_mean.shape[1]
            q_mean_real = q_mean[..., 0].to(dtype=torch.float32)
            q_mean_imag = q_mean[..., 1].to(dtype=torch.float32)
            if "q_abs_mean" in layer_data:
                q_abs_raw = layer_data["q_abs_mean"].to(dtype=torch.float32)
                if q_abs_raw.ndim == 1:
                    fitted_abs = _fit_1d(q_abs_raw, freq_count)
                else:
                    fitted_abs = torch.stack(
                        [_fit_1d(row, freq_count) for row in q_abs_raw],
                        dim=0,
                    )
                if fitted_abs.shape[0] == 1:
                    q_abs_mean = fitted_abs.expand(h, -1).contiguous()
                elif fitted_abs.shape[0] == h:
                    q_abs_mean = fitted_abs.contiguous()
                else:
                    raise ValueError(
                        f"layer {layer_idx}: q_abs_mean has "
                        f"{fitted_abs.shape[0]} rows, expected 1 or {h}"
                    )
            else:
                q_abs_mean = torch.sqrt(
                    q_mean_real**2 + q_mean_imag**2 + 1e-8
                )
            if "freq_scale_sq" in layer_data:
                fsq = layer_data["freq_scale_sq"].to(dtype=torch.float32)
                if fsq.ndim == 1:
                    fitted = _fit_1d(fsq, freq_count)
                else:
                    fitted = torch.stack(
                        [_fit_1d(row, freq_count) for row in fsq], dim=0
                    )
                if fitted.shape[0] == 1:
                    fsq = fitted.expand(h, -1).contiguous()
                elif fitted.shape[0] == h:
                    fsq = fitted.contiguous()
                else:
                    raise ValueError(
                        f"layer {layer_idx}: freq_scale_sq has "
                        f"{fitted.shape[0]} rows, expected 1 or {h}"
                    )
            else:
                fsq = torch.ones(h, freq_count)
            layers[int(layer_idx)] = LayerHeadStats(
                q_mean_real=q_mean_real.to(device),
                q_mean_imag=q_mean_imag.to(device),
                q_abs_mean=q_abs_mean.to(device),
                freq_scale_sq=fsq.to(device),
            )

    if not layers:
        raise ValueError("stats file contains no usable layer statistics")
    max_freq = max(ls.freq_count for ls in layers.values())
    omega = _inv_freq_from_metadata(metadata, max_freq, device)

    layer_omega: dict[int, torch.Tensor] = {}
    per_layer_freqs = metadata.get("layer_freq_counts")
    if isinstance(per_layer_freqs, dict):
        for layer_idx, freq_count in per_layer_freqs.items():
            layer_idx = int(layer_idx)
            if omega.numel() >= int(freq_count):
                layer_omega[layer_idx] = omega[: int(freq_count)].contiguous()

    head_dim = int(metadata.get("head_dim", max_freq * 2))
    return TriAttentionStats(
        head_dim=head_dim,
        rope_style=str(metadata.get("rope_style", "half")),
        omega=omega,
        layers=layers,
        layer_omega=layer_omega,
        num_kv_heads=int(metadata.get("num_kv_heads", 0)) or None,
    )


def score_layer_dense(
    k_rot: torch.Tensor,
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: int,
    *,
    rope_style: str = "half",
    score_aggregation: str = "mean",
    disable_trig: bool = False,
    disable_mlr: bool = False,
) -> torch.Tensor:
    """Score one layer's tokens from post-RoPE keys (K_rot-direct formula).

    Args:
        k_rot: [num_tokens, num_kv_heads, head_dim] post-RoPE keys.
        q_mean_real/q_mean_imag/q_abs_mean/freq_scale_sq: [num_kv_heads, F].
        omega: [F] rotation frequencies. offsets: [num_offsets] query
            offsets. round_start: absolute position of the decode round.

    Returns:
        [num_tokens, num_kv_heads] per-head importance scores.
    """
    if k_rot.ndim != 3:
        raise ValueError(
            f"k_rot must be [num_tokens, num_kv_heads, head_dim], "
            f"got shape {tuple(k_rot.shape)}"
        )
    n_tok, n_heads, head_dim = k_rot.shape
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    freq_count = head_dim // 2
    if omega.numel() < freq_count:
        raise ValueError(
            f"omega has {omega.numel()} frequencies, need {freq_count}"
        )
    omega = omega[:freq_count].to(device=k_rot.device, dtype=torch.float32)

    k = k_rot.to(dtype=torch.float32)
    if rope_style == "interleaved":
        pairs = k.reshape(n_tok, n_heads, freq_count, 2)
        k_real, k_imag = pairs[..., 0], pairs[..., 1]
    else:
        k_real, k_imag = k[..., :freq_count], k[..., freq_count:]

    q_mean_real = q_mean_real.to(device=k_rot.device, dtype=torch.float32)
    q_mean_imag = q_mean_imag.to(device=k_rot.device, dtype=torch.float32)
    q_abs_mean = q_abs_mean.to(device=k_rot.device, dtype=torch.float32)
    freq_scale_sq = freq_scale_sq.to(device=k_rot.device, dtype=torch.float32)

    # A + iB = Q_mean * conj(K_rot); rotation-invariant |K| magnitudes.
    prod_real = (
        q_mean_real.unsqueeze(0) * k_real + q_mean_imag.unsqueeze(0) * k_imag
    )
    prod_imag = (
        q_mean_imag.unsqueeze(0) * k_real - q_mean_real.unsqueeze(0) * k_imag
    )
    k_abs = torch.sqrt(k_real * k_real + k_imag * k_imag)

    scores_per_offset = []
    for offset_val in offsets.tolist():
        t = round_start + offset_val
        if disable_trig:
            scores_per_offset.append(torch.zeros(n_tok, n_heads, device=k.device))
            continue
        phase = t * omega  # [F]
        cos_v = torch.cos(phase).unsqueeze(0)  # [1, F]
        sin_v = torch.sin(phase).unsqueeze(0)
        term = freq_scale_sq * (prod_real * cos_v - prod_imag * sin_v)
        scores_per_offset.append(term.sum(dim=-1))  # [n_tok, n_heads]
    stacked = torch.stack(scores_per_offset, dim=0)  # [n_off, n_tok, H]
    if score_aggregation == "mean":
        scores = stacked.mean(dim=0)
    elif score_aggregation == "max":
        scores = stacked.max(dim=0).values
    else:
        raise ValueError(f"unsupported score_aggregation: {score_aggregation}")

    q_mean_abs = torch.sqrt(q_mean_real**2 + q_mean_imag**2 + 1e-8)
    extra_coef = q_abs_mean if disable_mlr else (q_abs_mean - q_mean_abs)
    extra_term = (k_abs * extra_coef.unsqueeze(0) * freq_scale_sq).sum(dim=-1)
    return scores + extra_term


class TriAttentionTokenScorer(TokenScorer):
    """TokenScorer backed by TriAttention frequency statistics.

    Aggregates per-(layer, kv-head) scores into one importance score per
    token.  All layers present in both the stats and the caller's gather are
    scored; layers missing from either side are skipped.
    """

    def __init__(
        self,
        stats_path: str | Path,
        device: torch.device,
        *,
        num_kv_heads: int | None = None,
        head_aggregation: str = "mean",
        score_aggregation: str = "mean",
        offsets_max_length: int = 65536,
        disable_trig: bool = False,
        disable_mlr: bool = False,
    ):
        if head_aggregation not in ("mean", "max"):
            raise ValueError("head_aggregation must be 'mean' or 'max'")
        if score_aggregation not in ("mean", "max"):
            raise ValueError("score_aggregation must be 'mean' or 'max'")
        self.stats = load_triattention_stats(
            stats_path, device, num_kv_heads=num_kv_heads
        )
        self.device = device
        self.head_aggregation = head_aggregation
        self.score_aggregation = score_aggregation
        self.disable_trig = disable_trig
        self.disable_mlr = disable_mlr
        self.offsets = build_geometric_offsets(offsets_max_length, device)
        # Per-device caches so the same scorer can score GPU gathers and
        # CPU offload-store reads without per-call host transfers.
        self._offsets_by_device: dict[torch.device, torch.Tensor] = {
            device: self.offsets
        }

    @property
    def scored_layers(self) -> list[int]:
        return sorted(self.stats.layers)

    @property
    def head_dim(self) -> int:
        return self.stats.head_dim

    def _offsets_on(self, device: torch.device) -> torch.Tensor:
        offsets = self._offsets_by_device.get(device)
        if offsets is None:
            offsets = self.offsets.to(device)
            self._offsets_by_device[device] = offsets
        return offsets

    def score(
        self,
        k_by_layer: Mapping[int, torch.Tensor],
        round_start: int,
    ) -> torch.Tensor:
        first = next(iter(k_by_layer.values()), None)
        if first is None or first.shape[0] == 0:
            dev = self.device
            return torch.zeros(0, device=dev, dtype=torch.float32)

        per_layer_scores: list[torch.Tensor] = []
        for layer_idx, layer_stats in self.stats.layers.items():
            k = k_by_layer.get(layer_idx)
            if k is None:
                continue
            if k.ndim != 3:
                raise ValueError(
                    f"layer {layer_idx}: k must be [n_tokens, num_kv_heads, "
                    f"head_dim], got {tuple(k.shape)}"
                )
            if k.shape[1] != layer_stats.num_kv_heads:
                raise ValueError(
                    f"layer {layer_idx}: expected {layer_stats.num_kv_heads} "
                    f"kv heads, got {k.shape[1]}"
                )
            if k.shape[2] != 2 * layer_stats.freq_count:
                raise ValueError(
                    f"layer {layer_idx}: expected head_dim "
                    f"{2 * layer_stats.freq_count}, got {k.shape[2]}"
                )
            omega = self.stats.layer_omega.get(layer_idx, self.stats.omega)
            layer_scores = score_layer_dense(
                k,
                layer_stats.q_mean_real,
                layer_stats.q_mean_imag,
                layer_stats.q_abs_mean,
                layer_stats.freq_scale_sq,
                omega,
                self._offsets_on(k.device),
                round_start,
                rope_style=self.stats.rope_style,
                score_aggregation=self.score_aggregation,
                disable_trig=self.disable_trig,
                disable_mlr=self.disable_mlr,
            )
            per_layer_scores.append(layer_scores)

        if not per_layer_scores:
            # No scored layer matched the gather; rank by magnitude of K=0.
            return torch.zeros(first.shape[0], device=first.device,
                              dtype=torch.float32)

        # [n_tokens, total_scored_heads]
        stacked = torch.cat(per_layer_scores, dim=1)
        if self.head_aggregation == "mean":
            return stacked.mean(dim=1)
        return stacked.max(dim=1).values
