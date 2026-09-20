# Persistent density sweep

## Objective

Reduce the 32K H200 density sweep from one vLLM server launch per
`density × offered concurrency` case to one launch per density, without
changing the kernel, mutating density inside a CUDA graph, or changing the
existing single-case benchmark command.

The target matrix is one dense baseline and BMM densities
`[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]`
across offered concurrencies `[64, 32, 16, 8, 4, 2, 1]`.

## Scope

- Add a reusable server-process session to `skylight.bench.run_serving`.
- Add a focused YAML-driven density-sweep runner.
- Start dense once and run every requested concurrency against it.
- Start sparse once per density and run every requested concurrency against it.
- Use CUDA graphs only and fix `max_num_seqs` to the largest requested concurrency.
- Warm each server on the first benchmark case only.
- Preserve independent result, client log, resolved workload, and status
  artifacts for every case, with server logs, versions, and telemetry owned by
  the server group.
- Resume by skipping successful cases and running only missing or failed cases.
- Emit one top-level JSON and CSV summary with dense-relative throughput and
  TPOT ratios.

## Non-goals

- Runtime topk mutation.
- New serving endpoints or engine RPCs.
- Kernel changes.
- A generic experiment framework.
- Multiple measurement trials.
- Changes to the active remote H200 sweep.
- Renaming or changing the signatures of existing public helpers in
  `skylight.bench.run_serving`.
- Changing the existing one-case `ArtifactPaths` layout.

## Architecture

`ServingSession` owns one vLLM server process and receives an already-built
serve command, sanitized environment, port, timeout, and optional log path:

1. Start the server in a new process group and capture its log.
2. Wait for `/health`.
3. Run any number of already-built `vllm bench serve` clients against the same
   port.
4. Verify the server is still alive before each client and after a failed
   client.
5. Terminate the process group on normal exit, benchmark failure, or exception.

Command construction and experiment policy remain outside the lifecycle class,
using the existing `run_serving` helpers. This keeps `ServingSession` small and
usable by future serving benchmarks without teaching it about density sweeps.

The existing `run_serving` CLI becomes a one-case caller of
`ServingSession`, preserving its arguments and behavior.

`skylight.bench.density_sweep` reads one YAML document, validates it into
immutable configuration objects, and creates server groups:

- group `dense`;
- one group for each BMM density.

Within a group, cases run in the YAML concurrency order. The server is
configured with `max_num_seqs = max(concurrencies)`, so vLLM captures all
required batch graph sizes during startup. Density remains immutable for the
entire sparse server lifetime because it changes selected-set sizes and
CUDA-graph structure.
Prefix caching remains disabled. Dense and every BMM density use the same
server limits, concurrency order, workload, generation parameters, and
hardware so the comparisons differ only in attention backend and BMM density.

The YAML name `concurrencies` means the number of prompts submitted at
`request_rate: inf`. It is the offered request concurrency and the server's
upper bound, not a claim that every decode step has that exact active batch
size. The vLLM result's `max_concurrent_requests` records the observed peak.

## Configuration

The checked-in experiment is `experiments/h200-32k-density.yaml`:

```yaml
model: Qwen/Qwen3.5-9B
model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
tokenizer_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
hardware:
  architecture: hopper
  expected_name: NVIDIA H200
  device: 0
  memory_utilization: 0.99
server:
  mode: cudagraph
  max_model_len: 32896
  dtype: bfloat16
  block_size: 16
  gdn_prefill_backend: triton
  timeout_seconds: 1800
  client_timeout_seconds: 1800
  hf_overrides:
    text_config:
      max_position_embeddings: 1048576
      rope_parameters:
        rope_type: yarn
        factor: 4.0
        original_max_position_embeddings: 262144
        rope_theta: 10000000
        partial_rotary_factor: 0.25
        mrope_interleaved: true
        mrope_section: [11, 11, 10]
workload:
  dataset: random
  context_length: 32768
  output_length: 128
  concurrencies: [64, 32, 16, 8, 4, 2, 1]
  request_rate: inf
  warmups: 2
  temperature: 0
  ignore_eos: true
bmm:
  method: block_minmax
  densities: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
  sink: 64
  local: 64
  channel_num: -1
```

The command is:

```bash
python -m skylight.bench.density_sweep \
  --config experiments/h200-32k-density.yaml \
  --output-root bench-results/h200-32k-density-fast
```

PyYAML is an explicit Skylight dependency rather than an undeclared
transitive dependency.

The server receives both immutable Hugging Face revisions. Because this
vLLM benchmark CLI does not expose a tokenizer-revision flag, the runner
resolves that exact tokenizer snapshot once and passes its local path with
`--tokenizer` to every client.

The focused runner rejects inherited `SKYLIGHT_TP`, `SKYLIGHT_KV_DTYPE`,
`SKYLIGHT_BENCH_PROFILE`, and `SKYLIGHT_BENCH_DETAILED` overrides. Other
allowlisted runtime environment settings are included in the run fingerprint.
Both Skylight repositories must have no relevant uncommitted source changes;
environment symlinks, result directories, `__pycache__`, and `.pyc` churn do
not count as source.

`hardware.architecture` uses Skylight's supported architecture names
(`hopper` or `blackwell`). `hardware.device` sets `CUDA_VISIBLE_DEVICES`
before runtime detection. `versions.json` records the exact detected GPU name,
UUID, driver, and compute capability, including whether the Hopper device is
an H100 or H200. The runner fails before server startup when the detected GPU
name does not contain the configured `expected_name`.

The 32K experiment deliberately uses `max_model_len: 32896`, exactly matching
the active 32K queue (`32768` input + `128` output). This is distinct from the
general long-context capacity probe, which adds 512 tokens of headroom.

Duplicate mapping keys and unknown keys are rejected at every YAML level,
except inside the intentionally open-ended `hf_overrides` payload.
Concurrencies and densities must be non-empty and unique; concurrencies and
token lengths must be positive; densities and memory utilization must be
finite and in `(0, 1]`; request rate must be positive or infinity;
`max_model_len` must be at least `context_length + output_length`; warmups must
be non-negative; server and client timeouts must be positive; and the only
accepted mode is `cudagraph`. Density path names use a canonical decimal
representation so the same density cannot acquire two artifact paths.

## Artifacts and resume

Artifacts use stable case paths:

```text
<output-root>/dense/server.log
<output-root>/dense/versions.json
<output-root>/dense/concurrency-64/result.json
<output-root>/topk-0.001/server.log
<output-root>/topk-0.001/telemetry/
<output-root>/topk-0.001/concurrency-64/result.json
<output-root>/topk-0.001/concurrency-32/result.json
```

Each case contains:

- `result.json`;
- `client.log`;
- `resolved.json`;
- `status.json`.

Each server group contains:

- `server.log`;
- `server-resolved.json`;
- `versions.json`;
- `telemetry/` for sparse groups;
- `group-status.json`.

Sparse telemetry includes an append-only `case-markers.jsonl` with each
concurrency's start/finish window so group-owned kernel metrics remain
attributable across resumed attempts. Case and group statuses preserve both
start and finish timestamps.

A case is complete only when `status.json` reports success, `result.json`
exists, `completed` equals the requested concurrency, and `failed` equals zero.
On resume, complete cases are skipped. If every case in a group is complete,
no server is started for that group.

Every group and case manifest contains:

- `schema_version: 1`;
- the canonical semantic configuration;
- a SHA-256 fingerprint of the configuration and immutable run identity.

The immutable run identity includes the Skylight and skylight-kernels
revisions and dirty flags, Python/package versions, CUDA/driver identity,
detected platform, and GPU name/UUID. It excludes timestamps. This prevents a
resume from silently combining results produced by different code, packages,
drivers, or physical GPUs.

Resume skips a case only when both its success criteria and fingerprint match.
A mismatched fingerprint fails before server startup and instructs the caller
to choose a new output root; there is no implicit overwrite. When rerunning a
matching incomplete or failed case, existing case artifacts move into a
numbered `attempts/` directory before the runner atomically writes
`status.json` with state `running`. Final statuses and summaries are also
written by temporary-file replacement so interruption cannot create a
half-written manifest.

Every incomplete case uses the configured client warmup count. The server,
model, kernel path, and CUDA graphs remain persistent, but client warmups are
kept identical to the legacy lifecycle so low-concurrency throughput remains
directly comparable.

After each case, the runner rewrites `summary.json` and `summary.csv`
atomically from all valid results already present under the output root. Each
BMM row contains its matching dense-concurrency metrics, output-throughput
ratio, and TPOT speedup. Summary rows require matching case fingerprints, and
both formats carry the run fingerprint; JSON also carries the semantic config.
This makes an interrupted run immediately plottable without losing provenance.

## Failure handling

- Configuration errors fail before starting a server.
- A server startup or health failure writes `group-status.json`, marks the
  group failed, and runs no clients in that group.
- A benchmark client failure writes `status.json`, leaves its logs intact,
  and continues to the next case only if `/health` still succeeds and the
  server process is alive.
- A benchmark client exceeding `client_timeout_seconds` is failed explicitly
  instead of inheriting vLLM's multi-hour HTTP timeout.
- If the server exits, remaining cases in that group are not attempted.
- `server.log` is opened in append mode with a timestamped attempt header, so
  resume never erases startup or crash evidence.
- Process-group teardown runs unconditionally.
- The runner exits nonzero if any requested case is incomplete or failed.

There are no automatic measurement retries.

## Tests

Unit tests cover:

- strict YAML parsing and validation;
- hardware selection occurs before runtime detection;
- stable group and artifact paths;
- canonical semantic fingerprints and mismatch rejection;
- stale failed artifacts are preserved under `attempts/`;
- one dense group plus one group per density;
- `max_num_seqs` equals the maximum concurrency;
- configured client warmups apply to every executed case in a group;
- completed cases are skipped without starting a server;
- one server session serves multiple benchmark clients;
- a failed client continues only while the server remains healthy;
- teardown occurs on success and exceptions;
- summaries pair each BMM result with the same-concurrency dense result;
- existing imported helper signatures and the one-case `ArtifactPaths` layout
  remain unchanged;
- the existing one-case `run_serving` behavior remains compatible.

Before fast results are treated as comparable to legacy results, the remote
H200 equivalence smoke runs dense and BMM topk `0.10` at concurrencies `64`
and `1` through both lifecycles with otherwise identical settings. All requests
must complete without failure, and persistent output throughput and mean TPOT
must each be within 5% of the legacy value for all four comparisons. This is a
single-trial compatibility gate, not a statistical performance claim. If it
fails, the fast matrix remains a separate matched experiment and is not merged
with legacy results until the cause is understood.
