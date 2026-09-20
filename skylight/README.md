# Skylight

Skylight is a vLLM sparse-attention backend for NVIDIA H100/H200 (SM90) and
B200 (SM100). The serving integration stays small; the CUDA kernels remain in
the separate `skylight-org/skylight_kernels` repository.

## Quickstart

```bash
git clone https://github.com/skylight-org/skylight.git
cd skylight
./setup.sh
```

Host prerequisites:

- Linux x86_64
- One selected H100, H200, or B200
- An NVIDIA driver compatible with CUDA 13
- `git`, `curl`, and a host C++ compiler
- Git authentication that can read `skylight-org/skylight_kernels`
- Enough operator-chosen model-cache space for `Qwen/Qwen3.5-9B`

`setup.sh` installs a pinned `uv`, Python 3.12 environment, CUDA 13 wheel
toolchain, vLLM stack, and Skylight kernels. Host Python and `uv` are not
prerequisites. To require a specific architecture:

```bash
./setup.sh --hardware hopper
./setup.sh --hardware blackwell
```

Setup clones the pinned kernels revision into `./skylight_kernels`, validates
that checkout, and compiles only for the detected architecture. Compilation
caches default to node-local `/var/tmp`; override them with
`SKYLIGHT_COMPILE_CACHE_DIR`. Setup does not change the Hugging Face or model
cache location. It is safe to rerun while the kernels checkout is clean and
still at the pinned revision.

If the repository is on a quota-limited network volume, keep the large Python
environment on node-local storage while preserving the `.venv` command path:

```bash
SKYLIGHT_ENV_DIR=/opt/skylight/venv ./setup.sh --hardware hopper
```

## Serving smoke tests

Dense control:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve Qwen/Qwen3.5-9B \
  --attention-backend FLASHINFER \
  --dtype bfloat16 \
  --max-model-len 17408 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.5 \
  --block-size 16 \
  --gdn-prefill-backend triton
```

Explicit block-minmax (BMM) sparse path:

```bash
SKYLIGHT_SPARSE_METHOD=block_minmax \
SKYLIGHT_SPARSE_TOPK=0.10 \
SKYLIGHT_SPARSE_SINK=64 \
SKYLIGHT_SPARSE_LOCAL=64 \
SKYLIGHT_SPARSE_CHANNEL_NUM=-1 \
SKYLIGHT_SPARSE_SUB_PAGE=16 \
SKYLIGHT_BLOCK_SIZE=16 \
SKYLIGHT_INCR_SLOT=1 \
SKYLIGHT_INCR_FULLCG=1 \
SKYLIGHT_INCR_PIPELINED=1 \
SKYLIGHT_FI_BSR=0 \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/skylight serve \
  --model Qwen/Qwen3.5-9B \
  --dtype bfloat16 \
  --max-model-len 17408 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.5 \
  --block-size 16 \
  --gdn-prefill-backend triton
```

The environment block is a temporary internal interface pending the compact
YAML configuration design. `SKYLIGHT_SPARSE_TOPK=0.10` is a nominal target
density, not the observed effective density.

## Release qualification

Run the same matched matrix on H100 and B200:

```bash
scripts/qualify_serving.sh --hardware hopper --gpu 0
scripts/qualify_serving.sh --hardware blackwell --gpu 0
```

Each command produces 12 conditions by default: dense and BMM, eager and CUDA
graph, at `B1/L4096/O64`, `B8/L4096/O64`, and `B8/L16384/O64`. Pass
`--trials 3` explicitly for a 36-condition stability run.
Every condition has its own `result.json`, server/client logs, resolved
settings, version manifest, and sparse telemetry directory under
`bench-results/qualification/<timestamp>/<H100-or-B200>/`.

For the fast qualification, compare the paired dense/BMM results directly. For
a stability run, compare paired medians across the three trials. Use the
telemetry—not the nominal target density—to report observed effective density.
Inspect the fast matrix without a GPU or environment:

```bash
scripts/qualify_serving.sh \
  --hardware blackwell \
  --gpu 0 \
  --dry-run
```

### Persistent H200 density sweep

Run the checked CUDA-graph-only 32K matrix with one dense server and one BMM
server per density:

```bash
.venv/bin/python -m skylight.bench.density_sweep \
  --config experiments/h200-32k-density.yaml \
  --output-root bench-results/h200-32k-density-fast
```

The runner executes all seven concurrencies against each persistent server,
reducing the 91-case matrix to 13 server launches. Successful cases resume
without relaunch; a code, package, driver, GPU, or config mismatch requires a
new output root. It requires committed Skylight/kernel source and rejects
benchmark overrides outside the YAML. Model and tokenizer commit revisions
are pinned; the benchmark uses the resolved tokenizer snapshot. Each client
has the checked 30-minute deadline, and a timeout tears down the tainted
server group before any later case runs. Follow progress in the terminal and
read the incrementally updated `summary.json` or `summary.csv`.

## Troubleshooting

- **Unsupported or mismatched GPU:** setup and qualification accept only
  H100/H200 (SM90) and B200 (SM100), and fail if `--hardware` disagrees with
  the visible device.
- **Kernels clone fails:** configure Git authentication for the private
  `skylight-org/skylight_kernels` repository, then rerun setup.
- **Kernels revision or dirty-check failure:** restore a clean checkout at the
  revision printed by setup. Setup will not overwrite local kernel changes.
- **Model download fails:** choose a model-cache volume with enough free space;
  Skylight deliberately does not relocate it.
- **Slow or unstable compilation:** do not point
  `SKYLIGHT_COMPILE_CACHE_DIR` at NFS; keep compilation caches node-local.
- **Invalid benchmark evidence:** use an otherwise idle selected GPU.
- **Hopper Qwen3.5 startup fails:** the qualified H100/H200 path requires
  `--gdn-prefill-backend triton`; do not change this for release comparison.

## More documentation

- [Docker and agentic benchmark](docs/docker.md)
- [Long-context benchmark](docs/longctx.md)
- [Design specifications](docs/specs/)

## License

Apache-2.0. See [LICENSE](LICENSE).
