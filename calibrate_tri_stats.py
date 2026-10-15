#!/usr/bin/env python3
"""Calibrate TriAttention Q statistics for a HF Llama-family model.

Collects post-RoPE query vectors per (layer, attention head) by hooking
``q_proj`` and applying the model's own rotary embedding at prompt
positions, then GQA-averages the per-head means onto KV heads and writes
the TriAttention runtime stats format consumed by
``offload_package.scoring.triattention.load_triattention_stats``:

    {
      'metadata': {num_attention_heads, num_kv_heads, head_dim, num_layers,
                   rope_theta, rope_style},
      'layer_stats': {layer: {
          'q_mean_complex': [num_kv_heads, freq_count, 2],
          'q_abs_mean':    [num_kv_heads, freq_count],
          'freq_scale_sq': [num_kv_heads, freq_count],   # ones (plain RoPE)
      }},
    }

Usage: .venv/bin/python calibrate_tri_stats.py [model] [out_path] [n_prompts]
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'stats/tinyllama_tri_stats.pt')
N_PROMPTS = int(sys.argv[3]) if len(sys.argv) > 3 else 24

PROMPTS = [
    f"Observation {i}: the calibrated reading was {(i * 37) % 997}, which "
    f"the field team called noteworthy, and the reviewers of section "
    f"{(i * 11) % 23} marked it pending further review. "
    for i in range(N_PROMPTS)
]

acc = None  # layer -> {'sum_c': [H, F] complex, 'sum_abs': [H, F], 'n': int}
handles = []


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    model.to('cuda')
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_attn = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_attn)
    head_dim = cfg.hidden_size // n_attn
    gqa = n_attn // n_kv
    freq = head_dim // 2

    acc = {
        layer: {
            'sum_c': torch.zeros(n_attn, freq, dtype=torch.complex128,
                                 device='cuda'),
            'sum_abs': torch.zeros(n_attn, freq, dtype=torch.float64,
                                   device='cuda'),
            'n': 0,
        }
        for layer in range(n_layers)
    }

    rotary = model.model.rotary_emb
    rotary_dtype = next(rotary.parameters()).dtype if any(
        True for _ in rotary.parameters()
    ) else torch.float32

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # output: [B, L, H*D] pre-RoPE queries for this layer.
            q = output.float().view(output.shape[0], output.shape[1],
                                    n_attn, head_dim)
            b, L = q.shape[0], q.shape[1]
            pos = torch.arange(L, device=q.device).view(1, L).expand(b, L)
            probe = torch.zeros(1, 1, head_dim, device=q.device,
                                dtype=rotary_dtype)
            cos, sin = rotary(probe, pos)
            # HF returns [B, L, head_dim]; broadcast over heads.
            q_rot = q * cos.unsqueeze(2) + rotate_half(q) * sin.unsqueeze(2)
            qc = torch.complex(q_rot[..., :freq], q_rot[..., freq:])
            entry = acc[layer_idx]
            entry['sum_c'] += qc.sum(dim=(0, 1)).double()
            entry['sum_abs'] += qc.abs().sum(dim=(0, 1)).double()
            entry['n'] += b * L
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(i)
        ))

    for p in PROMPTS:
        ids = tok(p, return_tensors='pt').input_ids.cuda()
        model(ids)
    torch.cuda.synchronize()
    for h in handles:
        h.remove()

    layer_stats = {}
    for layer in range(n_layers):
        e = acc[layer]
        mean_c = (e['sum_c'] / e['n']).reshape(n_kv, gqa, freq).mean(dim=1)
        mean_abs = (e['sum_abs'] / e['n']).reshape(n_kv, gqa, freq).mean(dim=1)
        q_mean_complex = torch.stack(
            [mean_c.real.float(), mean_c.imag.float()], dim=-1
        )  # [n_kv, freq, 2]
        layer_stats[layer] = {
            'q_mean_complex': q_mean_complex.cpu(),
            'q_abs_mean': mean_abs.float().cpu(),
            'freq_scale_sq': torch.ones(n_kv, freq),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'metadata': {
                'num_attention_heads': n_attn,
                'num_kv_heads': n_kv,
                'head_dim': head_dim,
                'num_layers': n_layers,
                'rope_theta': float(getattr(cfg, 'rope_theta', 10000.0)),
                'rope_style': 'half',
                'model_name': MODEL,
            },
            'layer_stats': layer_stats,
        },
        OUT,
    )

    # Sanity: means should be non-degenerate (|q_mean| <= q_abs_mean, both
    # nonzero, and the MLR gap positive on average).
    m0 = layer_stats[0]
    mag = torch.stack(
        [m0['q_mean_complex'][..., 0], m0['q_mean_complex'][..., 1]], -1
    ).norm(dim=-1)
    print(f"wrote {OUT} (layers={n_layers}, kv_heads={n_kv}, "
          f"head_dim={head_dim}, tokens~={acc[0]['n']})")
    print(f"layer0 |q_mean| mean={mag.mean():.4f}  "
          f"q_abs mean={m0['q_abs_mean'].mean():.4f}  "
          f"mlr gap={(m0['q_abs_mean'] - mag).mean():.4f}")


if __name__ == '__main__':
    main()
