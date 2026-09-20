#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE=auto
UV_VERSION=0.11.28
KERNELS_SHA=b88bd0f4758ebb90cd388786af3e4f4fbd9c3cde
KERNELS_URL="${SKYLIGHT_KERNELS_URL:-https://github.com/skylight-org/skylight_kernels.git}"
KERNELS="$ROOT/skylight_kernels"
UV_DIR="$ROOT/.skylight/bin"
UV="$UV_DIR/uv"
VENV="${SKYLIGHT_ENV_DIR:-$ROOT/.venv}"
export UV_PROJECT_ENVIRONMENT="$VENV"

usage() {
  printf '%s\n' \
    "Usage: ./setup.sh [--hardware auto|hopper|blackwell]" \
    "" \
    "Build the pinned Skylight environment for one visible H100 or B200." \
    "" \
    "Environment overrides:" \
    "  SKYLIGHT_KERNELS_URL         Git URL for the separate kernels checkout" \
    "  SKYLIGHT_ENV_DIR             Python environment directory (default: .venv)" \
    "  SKYLIGHT_COMPILE_CACHE_DIR   Node-local extension/Inductor cache root"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --hardware)
      if [ "$#" -lt 2 ]; then
        printf 'setup: --hardware requires auto, hopper, or blackwell\n' >&2
        exit 2
      fi
      HARDWARE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'setup: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$HARDWARE" in
  auto|hopper|blackwell) ;;
  *)
    printf 'setup: --hardware must be auto, hopper, or blackwell; got %s\n' \
      "$HARDWARE" >&2
    exit 2
    ;;
esac

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  printf 'setup: requires Linux x86_64\n' >&2
  exit 1
fi

for command in git curl nvidia-smi; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'setup: missing host prerequisite: %s\n' "$command" >&2
    exit 1
  fi
done
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  printf 'setup: missing host prerequisite: c++ or g++\n' >&2
  exit 1
fi

mkdir -p "$UV_DIR"
if [ ! -x "$UV" ]; then
  printf 'setup: installing uv %s into %s\n' "$UV_VERSION" "$UV_DIR"
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" |
    env UV_UNMANAGED_INSTALL="$UV_DIR" sh
fi

if [ ! -e "$KERNELS" ]; then
  printf 'setup: cloning kernels from %s\n' "$KERNELS_URL"
  git clone --no-checkout "$KERNELS_URL" "$KERNELS"
  git -C "$KERNELS" checkout --detach "$KERNELS_SHA"
elif ! git -C "$KERNELS" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'setup: %s exists but is not a Git checkout\n' "$KERNELS" >&2
  exit 1
fi

if [ -n "$(git -C "$KERNELS" status --porcelain)" ]; then
  printf 'setup: kernels checkout is dirty; refusing to overwrite %s\n' \
    "$KERNELS" >&2
  exit 1
fi
ACTUAL_KERNELS_SHA="$(git -C "$KERNELS" rev-parse HEAD)"
if [ "$ACTUAL_KERNELS_SHA" != "$KERNELS_SHA" ]; then
  printf 'setup: kernels revision mismatch\n' >&2
  printf '  actual:   %s\n' "$ACTUAL_KERNELS_SHA" >&2
  printf '  required: %s\n' "$KERNELS_SHA" >&2
  exit 1
fi

if [ "$VENV" != "$ROOT/.venv" ]; then
  mkdir -p "$(dirname -- "$VENV")"
  if [ -L "$ROOT/.venv" ]; then
    if [ "$(readlink "$ROOT/.venv")" != "$VENV" ]; then
      printf 'setup: .venv points somewhere other than SKYLIGHT_ENV_DIR\n' >&2
      exit 1
    fi
  elif [ -e "$ROOT/.venv" ]; then
    printf 'setup: .venv exists; move it before setting SKYLIGHT_ENV_DIR\n' >&2
    exit 1
  else
    ln -s "$VENV" "$ROOT/.venv"
  fi
fi

PYTHON="$VENV/bin/python"
if [ ! -x "$PYTHON" ] ||
  ! "$PYTHON" -c "import skylight.runtime, torch" >/dev/null 2>&1; then
  printf 'setup: bootstrapping the frozen Python 3.12 environment\n'
  "$UV" sync \
    --frozen \
    --python 3.12 \
    --no-install-package skylight-kernels
fi

RUNTIME_ENV="$ROOT/.skylight/runtime.env"
configure_runtime() {
  "$PYTHON" -m skylight.runtime \
    --hardware "$HARDWARE" \
    --write-env "$RUNTIME_ENV"
  set -a
  . "$RUNTIME_ENV"
  set +a
}
configure_runtime

printf 'setup: syncing the pinned workspace for %s (%s)\n' \
  "$SKYLIGHT_PLATFORM" "$TORCH_CUDA_ARCH_LIST"
"$UV" sync \
  --frozen \
  --python 3.12 \
  --no-build-isolation-package skylight-kernels
configure_runtime

(
  cd "$KERNELS"
  "$PYTHON" -m pytest -q test_block_minmax_optimized.py
  "$PYTHON" -m pytest -q test_incremental_summary_cache.py
  "$PYTHON" -m pytest --import-mode=importlib -q test_incr_parity.py
)

"$PYTHON" -c \
  "import skylight; from block_minmax_incr_optimized import BatchDecodeWithPagedKVCacheWrapper; print('imports: ok')"

case "$SKYLIGHT_PLATFORM" in
  hopper) SM=sm90 ;;
  blackwell) SM=sm100 ;;
esac
SKYLIGHT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

printf '%s\n' \
  "" \
  "Skylight setup complete" \
  "  platform:      $SKYLIGHT_PLATFORM" \
  "  architecture:  $SM" \
  "  skylight:      $SKYLIGHT_SHA" \
  "  kernels:       $ACTUAL_KERNELS_SHA" \
  "  CUDA_HOME:     $CUDA_HOME" \
  "  compile cache: $SKYLIGHT_COMPILE_CACHE_DIR" \
  "  executable:    $ROOT/.venv/bin/skylight" \
  "" \
  "Next:" \
  "  $ROOT/.venv/bin/skylight serve --model Qwen/Qwen3.5-9B"
