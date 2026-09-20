#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
HARDWARE=
GPU=
TRIALS=1
OUTPUT_ROOT=
DRY_RUN=0

usage() {
  printf '%s\n' \
    "Usage: scripts/qualify_serving.sh --hardware hopper|blackwell --gpu N [options]" \
    "" \
    "Options:" \
    "  --trials N          Paired repetitions (default: 1)" \
    "  --output-root PATH  Artifact root (default: bench-results/qualification/<UTC>)" \
    "  --dry-run           Print the complete matrix without requiring a GPU or venv" \
    "  --help              Show this help"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --hardware)
      [ "$#" -ge 2 ] || {
        printf 'qualify: --hardware requires a value\n' >&2
        exit 2
      }
      HARDWARE="$2"
      shift 2
      ;;
    --gpu)
      [ "$#" -ge 2 ] || {
        printf 'qualify: --gpu requires a value\n' >&2
        exit 2
      }
      GPU="$2"
      shift 2
      ;;
    --trials)
      [ "$#" -ge 2 ] || {
        printf 'qualify: --trials requires a value\n' >&2
        exit 2
      }
      TRIALS="$2"
      shift 2
      ;;
    --output-root)
      [ "$#" -ge 2 ] || {
        printf 'qualify: --output-root requires a value\n' >&2
        exit 2
      }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'qualify: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$HARDWARE" in
  hopper) HARDWARE_LABEL=H100 ;;
  blackwell) HARDWARE_LABEL=B200 ;;
  *)
    printf 'qualify: --hardware must be hopper or blackwell\n' >&2
    exit 2
    ;;
esac
case "$GPU" in
  ''|*[!0-9]*)
    printf 'qualify: --gpu must be a non-negative integer\n' >&2
    exit 2
    ;;
esac
case "$TRIALS" in
  ''|*[!0-9]*|0)
    printf 'qualify: --trials must be a positive integer\n' >&2
    exit 2
    ;;
esac

if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$ROOT/bench-results/qualification/$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
fi
ARTIFACT_ROOT="$OUTPUT_ROOT/$HARDWARE_LABEL"

if [ "$DRY_RUN" -eq 0 ]; then
  if [ ! -x "$PYTHON" ]; then
    printf 'qualify: missing %s; run ./setup.sh first\n' "$PYTHON" >&2
    exit 1
  fi
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf 'qualify: missing nvidia-smi\n' >&2
    exit 1
  }

  mkdir -p "$ARTIFACT_ROOT"
  env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" -m skylight.runtime \
      --hardware "$HARDWARE" \
      --print-json \
      >"$ARTIFACT_ROOT/runtime.json"
  nvidia-smi -q >"$ARTIFACT_ROOT/nvidia-smi-q.txt"
  {
    printf 'hardware=%q\n' "$HARDWARE"
    printf 'hardware_label=%q\n' "$HARDWARE_LABEL"
    printf 'gpu=%q\n' "$GPU"
    printf 'trials=%q\n' "$TRIALS"
    printf 'output_root=%q\n' "$OUTPUT_ROOT"
    printf 'model=%q\n' "Qwen/Qwen3.5-9B"
  } >"$ARTIFACT_ROOT/qualification.env"
fi

run_condition() {
  local trial="$1"
  local mode="$2"
  local backend="$3"
  local batch="$4"
  local input_len="$5"
  local backend_label="$backend"
  local mode_flag=--enforce-eager
  local artifacts
  local -a command

  if [ "$backend" = sparse ]; then
    backend_label=bmm
  fi
  if [ "$mode" = cudagraph ]; then
    mode_flag=--no-enforce-eager
  fi
  artifacts="$ARTIFACT_ROOT/trial-$(printf '%02d' "$trial")/$mode/${backend_label}-B${batch}-L${input_len}-O64"

  command=(
    env "CUDA_VISIBLE_DEVICES=$GPU"
    "$PYTHON" -m skylight.bench.run_serving
    --backend "$backend"
    --model Qwen/Qwen3.5-9B
    --dataset-name random
    --num-prompts "$batch"
    --random-input-len "$input_len"
    --random-output-len 64
    --request-rate inf
    --max-model-len 17408
    --max-num-seqs 8
    --gpu-memory-utilization 0.5
    --gdn-prefill-backend triton
    --dtype bfloat16
    --block-size 16
    --num-warmups 2
    --temperature 0
    --ignore-eos
    --server-timeout 1800
    "$mode_flag"
    --artifacts-dir "$artifacts"
  )
  if [ "$backend" = sparse ]; then
    command+=(
      --method block_minmax
      --topk 0.10
      --sink 64
      --local 64
      --channel-num -1
    )
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    printf 'qualify: trial=%02d mode=%s backend=%s B=%s L=%s\n' \
      "$trial" "$mode" "$backend_label" "$batch" "$input_len"
    "${command[@]}"
  fi
}

for ((trial = 1; trial <= TRIALS; trial++)); do
  for mode in eager cudagraph; do
    for workload in "1 4096" "8 4096" "8 16384"; do
      read -r batch input_len <<<"$workload"
      run_condition "$trial" "$mode" dense "$batch" "$input_len"
      run_condition "$trial" "$mode" sparse "$batch" "$input_len"
    done
  done
done
