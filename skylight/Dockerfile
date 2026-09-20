# Dockerfile for skylight on NVIDIA B200 (sm_100).
#
# Encodes the actual working setup discovered by running the bench
# end-to-end (cu13 toolchain pin set, wheel-layout symlinks, kernel
# native build).
#
# Build context expects skylight/ and skylight_kernels/ as siblings:
#   cd /path/to/parent && docker build -f skylight/Dockerfile -t skylight:dev .
#
# Run (B200 host with nvidia-container-toolkit):
#   docker run --rm -it --gpus all --shm-size=16g \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     skylight:dev

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# Minimal apt deps + Python 3.12 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg software-properties-common \
        build-essential ninja-build docker.io \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && cp /root/.local/bin/uv /usr/local/bin/uv

# Pin both repos to specific commits at build time. Pass via:
#   docker build \
#     --build-arg SKYLIGHT_SHA=$(git -C skylight rev-parse HEAD) \
#     --build-arg SKYLIGHT_KERNELS_SHA=$(git -C skylight_kernels rev-parse HEAD) \
#     ...
# Defaults to "unknown" so a build without --build-arg still works but
# the image's SHA labels show "unknown" (a clear "not pinned" marker).
ARG SKYLIGHT_SHA=unknown
ARG SKYLIGHT_KERNELS_SHA=unknown

# Project source (build context = parent of both repos, mounted as
# siblings; mirrors the on-host layout the uv workspace expects).
COPY ./skylight         /opt/skylight
COPY ./skylight_kernels /opt/skylight_kernels

# The pyproject's uv workspace member resolves through a sibling
# symlink. Re-create it inside the image so uv sync works.
WORKDIR /opt/skylight
RUN ln -sfn /opt/skylight_kernels skylight_kernels

# Stamp the SHAs into both the image (LABEL for `docker inspect`) and
# the filesystem (so runtime code can `cat /opt/skylight/.sha`).
RUN echo "$SKYLIGHT_SHA"         > /opt/skylight/.skylight-sha \
 && echo "$SKYLIGHT_KERNELS_SHA" > /opt/skylight/.skylight-kernels-sha
LABEL skylight.sha="${SKYLIGHT_SHA}"
LABEL skylight.kernels.sha="${SKYLIGHT_KERNELS_SHA}"

# Resolve everything EXCEPT skylight-kernels (which we build from
# source below with the pinned cu13 toolchain). --clear in case the
# build context accidentally included a host .venv.
RUN uv venv --python 3.12 --clear \
 && uv sync --extra agentic --extra longctx --no-install-package skylight-kernels

# Pin cu13 sub-packages exactly to the working venv on vb200-06.
# Without the explicit pin, the resolver mixes 13.0.x runtime (pulled
# transitively by cuda-toolkit) with whatever the resolver picks for
# nvcc/cccl/crt/nvvm. That mismatch triggers the "CUDA compiler and
# CUDA toolkit headers are incompatible" build error.
RUN uv pip install --no-deps \
        'nvidia-cuda-nvcc==13.0.88' \
        'nvidia-cuda-cccl==13.0.85' \
        'nvidia-cuda-crt==13.0.88' \
        'nvidia-cuda-runtime==13.0.96' \
        'nvidia-cuda-cupti==13.0.85' \
        'nvidia-cuda-nvrtc==13.0.88' \
        'nvidia-nvjitlink==13.0.88' \
        'nvidia-nvvm==13.0.88' \
        'nvidia-nvtx==13.0.85'

# Wheel-layout fixes (3 issues in the pip-shipped layout):
#   lib/   is the only dir; the linker (-L .../lib64) needs lib64
#   libcuda.so stub is missing from lib/stubs/ for build-time linking
#   versioned .so.13 names lack the unversioned symlink the linker wants
RUN set -eux; \
    CU13=/opt/skylight/.venv/lib/python3.12/site-packages/nvidia/cu13 && \
    ln -sfn lib "$CU13/lib64" && \
    mkdir -p "$CU13/lib/stubs" && \
    ln -sfn /usr/local/cuda/lib64/stubs/libcuda.so "$CU13/lib/stubs/libcuda.so" && \
    cd "$CU13/lib" && \
    for f in lib*.so.*; do \
        base="${f%%.so.*}"; \
        [ ! -e "${base}.so" ] && ln -s "$f" "${base}.so" || true; \
    done

# Project env.sh (also written into the image; entrypoint sources it)
RUN printf '%s\n' \
    'export CUDA_HOME=/opt/skylight/.venv/lib/python3.12/site-packages/nvidia/cu13' \
    'export PATH="$CUDA_HOME/bin:$PATH"' \
    > /opt/skylight/env.sh

# Build native kernel extensions + editable install both packages.
# This is the slow step (~5-10 min); cached on repeat builds.
# TORCH_CUDA_ARCH_LIST is required during docker build because the
# builder has no GPU; torch can't auto-detect arch from nvidia-smi.
# "10.0" = sm_100 = B200. Add other arches space-separated if needed.
ENV TORCH_CUDA_ARCH_LIST="10.0"
RUN bash -c '. /opt/skylight/env.sh && \
    cd /opt/skylight_kernels && \
    /opt/skylight/.venv/bin/python setup.py build_ext --inplace && \
    uv pip install --python /opt/skylight/.venv/bin/python --no-build-isolation --no-deps -e /opt/skylight_kernels && \
    uv pip install --python /opt/skylight/.venv/bin/python --no-build-isolation --no-deps -e /opt/skylight'

# Build-time smoke (fails the image build if imports break)
RUN bash -c '. /opt/skylight/env.sh && \
    /opt/skylight/.venv/bin/python -c "import skylight, skylight_kernels; print(\"OK\")" && \
    /opt/skylight/.venv/bin/python -c "from sparse_oracle_topk_sink_local_optimized import _cuda_ops; print(\"OK kernel ext\")"'

# LongBench/RULER/InfiniteBench scorers tokenize via nltk punkt.
RUN /opt/skylight/.venv/bin/python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab
ENV NLTK_DATA=/usr/share/nltk_data

# HF cache as volume so weights persist across container restarts
ENV HF_HOME=/root/.cache/huggingface
VOLUME /root/.cache/huggingface

# Entrypoint sources env.sh + activates the venv before exec'ing
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'source /opt/skylight/env.sh' \
    'source /opt/skylight/.venv/bin/activate' \
    'exec "$@"' \
    > /usr/local/bin/skylight-entry \
 && chmod +x /usr/local/bin/skylight-entry

ENTRYPOINT ["/usr/local/bin/skylight-entry"]
CMD ["bash"]
