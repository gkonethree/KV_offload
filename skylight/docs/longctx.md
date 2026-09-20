# Long-context benchmark lane

Run the long-context accuracy benchmarks (LongBench, LongBench-v2, RULER,
InfiniteBench, Loogle) against a skylight server (dense or sparse), the same
way the SWE-bench lane works: launch the server, run the benchmark against its
OpenAI-compatible API, score, write `summary.json`.

The benchmark definitions (datasets + prompts + scoring) are vendored from
`sparse-attention-hub` under `src/skylight/bench/longctx/`; generation is routed
to the server by `ApiServerAdapter` (HTTP, token-exact prompt build mirroring the
hub's HF adapter). Sparsity lives in the server, so the adapter is method-agnostic.

## Setup

Deps come from the `longctx` extra (already in the Dockerfile build):

```bash
uv sync --extra longctx                       # rouge, bert-score, fuzzywuzzy, jieba, nltk
python -m nltk.downloader punkt punkt_tab     # scorers tokenize via nltk
```

The sparse server imports its kernel from the `skylight_kernels` tree. In the
Docker image it is editable-installed (importable directly); on a host build that
used `build_ext --inplace`, the runner prepends the tree to the server's
`PYTHONPATH` automatically (override via `SKYLIGHT_KERNELS_PATH`).

## Run

```bash
python -m skylight.bench.longctx.run \
  --benchmark longbench --backend sparse --method block_minmax \
  --model mistralai/Ministral-3-14B-Instruct-2512 \
  --subsets narrativeqa --max-context 32768 --max-new-tokens 64 \
  --gpu 0 --gpu-mem 0.5 --out ./res/longbench_bmm
```

Writes `summary.json` (+ `metrics.json`, raw CSV, `config.json`) under `--out`,
and prints a `RESULT {...}` line with the per-task scores.

## Knobs

| flag | meaning |
|---|---|
| `--benchmark` | one of `longbench longbenchv2 ruler infinite_bench loogle` |
| `--backend`   | `dense` or `sparse` |
| `--method`    | `block_minmax` or `oracle` (sparse only) |
| `--topk / --sink / --local / --channel-num` | sparse selection knobs (forwarded to `SKYLIGHT_SPARSE_*`) |
| `--max-context` | token budget for the context (truncated per request) |
| `--gpu-mem`   | vLLM `gpu_memory_utilization`; lower it on a contended GPU |

Notes:
- The sparse summary buffer scales with `--max-context`; on a memory-tight GPU,
  reduce context and/or `--gpu-mem`.
- `--subsets` defaults to all datasets for the benchmark; pass a comma-separated
  subset for a quick run.
