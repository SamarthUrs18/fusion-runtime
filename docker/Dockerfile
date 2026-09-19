# syntax=docker/dockerfile:1
#
# fusion-runtime on an NVIDIA GPU.
#
#   docker build -f docker/Dockerfile -t fusion-runtime .
#   docker run --gpus all -p 8000:8000 -v /models:/models \
#     -e FUSION_ACCEPTED_KEYS=... -e FUSION_TRUSTED_PROXY=... fusion-runtime
#
# Two things this deliberately does not do: it installs no models (they live on a
# volume, so a rebuild doesn't re-download several gigabytes and two pods can
# share one copy), and it installs dependencies from pyproject.toml rather than a
# hand-written list, which is how the previous version of this file silently fell
# behind the package.
#
# Built for linux/amd64: GPU builds of torch and llama.cpp are published for that
# platform only. On an arm64 machine use Dockerfile.cpu, or let CI build this one.

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# torch comes from PyPI: its Linux wheels already carry CUDA and bundle the
# nvidia-* libraries. The cu124 index is not a "GPU version" of them — it is an
# older CUDA line that stops at torch 2.6, which is why pinning 2.9.1 against it
# failed. onnxruntime is the one that genuinely needs a different package name.
#
# llama-cpp-python is the opposite case: PyPI ships source only, and a plain
# install builds it without CUDA. Prebuilt CUDA wheels are tried first, and
# failing that it is compiled here, where nvcc exists.
RUN pip install torch==2.9.1 torchaudio==2.9.1
# CUDA_ARCHITECTURES is not optional here: left alone, nvcc builds kernels for
# every architecture it knows and the compile takes over an hour. These four
# cover the cards this actually runs on — 80 A100, 86 A10/A4000/3090,
# 89 L4/L40S/4090, 90 H100. A card outside the list still works through PTX
# just-in-time compilation, only slower to start.
ARG CUDA_ARCHITECTURES="80;86;89;90"
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    || CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
       pip install --no-cache-dir --verbose llama-cpp-python
RUN pip install onnxruntime-gpu

# Only what the install needs, so editing source doesn't invalidate the layers above.
COPY pyproject.toml README.md ./
COPY fusion_runtime/ ./fusion_runtime/
RUN pip install .

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FUSION_MODEL_DIR=/models \
    FUSION_CONFIG=production \
    FUSION_LOG_FORMAT=json

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 fusion \
    && mkdir -p /models && chown fusion:fusion /models

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER fusion

EXPOSE 8000

# /health is the one endpoint that needs no key, precisely so this works.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# An agent file is named by FUSION_AGENT (mount it, or bake it in); without one
# the production profile is used. 0.0.0.0 is refused unless keys are configured,
# which is the point.
ENTRYPOINT ["frun"]
CMD ["up", "--host", "0.0.0.0", "--port", "8000"]
