# syntax=docker/dockerfile:1
#
# fusion-runtime on an NVIDIA GPU: Whisper and Kokoro on CUDA, the LLM on vLLM or
# SGLang beside it (see docker/docker-compose.yml), llama.cpp for the CPU.
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

ARG CUDA_VERSION=12.8.1
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
# llama-cpp-python: PyPI ships source only, so it is compiled here either way.
#
# By default it is built for the CPU. On a GPU the LLM belongs on vLLM or SGLang
# (LLM("vllm:...")), which serves about three times the callers of in-process
# llama.cpp, and compiling llama.cpp's CUDA kernels took over 90 minutes on a
# two-core CI runner, so the image was never published. The CPU build takes a few
# minutes and stays useful for small models and tests.
#
#   docker build --build-arg LLAMA_CUDA=1 -f docker/Dockerfile .   # in-process llama.cpp on the GPU
#
# CUDA_ARCHITECTURES then decides the compile cost: 89-real is native code for
# L4, L40S and 4090; 80-virtual leaves PTX the driver compiles at load time for
# anything newer (A100, A10, H100). GGML_NATIVE=OFF either way, or the build is
# tuned to the build machine's CPU and dies with "illegal instruction" elsewhere.
ARG LLAMA_CUDA=0
ARG CUDA_ARCHITECTURES="80-virtual;89-real"
RUN if [ "$LLAMA_CUDA" = "1" ]; then \
        CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
            pip install --no-cache-dir llama-cpp-python; \
    else \
        CMAKE_ARGS="-DGGML_CUDA=off -DGGML_NATIVE=OFF" pip install --no-cache-dir llama-cpp-python; \
    fi
# From the CUDA 12 feed: PyPI's current onnxruntime-gpu is built for CUDA 13 and
# cannot load beside torch's 12.8, so it quietly runs on the CPU instead.
RUN pip install onnxruntime-gpu \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Only what the install needs, so editing source doesn't invalidate the layers above.
#
# Its own folder: without a WORKDIR these files landed in /, and `pip install .`
# then had setuptools look for fusion's packages across the whole filesystem
# (/proc, /usr, the multi-GB venv). The build sat there for an hour until CI
# cancelled it. LICENSE and NOTICE go in the package (Apache-2.0 asks for NOTICE).
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY fusion_runtime/ ./fusion_runtime/
RUN pip install .
# fusion's package asks for `onnxruntime`, and pip installs the CPU build of it beside
# onnxruntime-gpu: both own the same module, the CPU one lands last, and Kokoro runs
# on the CPU without a word (pod test, 25 September). Take it out, put the GPU build
# back, and fail the build if the CUDA provider isn't compiled in.
RUN pip uninstall -y onnxruntime \
    && pip install --force-reinstall --no-deps onnxruntime-gpu \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    && python -c "import onnxruntime as o; p = o.get_available_providers(); assert 'CUDAExecutionProvider' in p, p"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime
ARG PYTHON_VERSION

# torch ships its own CUDA libraries and expects them first. With the image's
# toolkit ahead of them, whichever library loads first wins: Whisper pulls in the
# container's older libcudart, torch then imports against it and dies on a
# missing symbol — taking voice detection down with it, so turns end on a timer
# and interruptions stop working. Found on the first GPU deployment.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}" \
    FUSION_MODEL_DIR=/models \
    FUSION_AGENT=/app/agent.py \
    FUSION_IMAGE=gpu \
    FUSION_LOG_FORMAT=json

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 fusion \
    && mkdir -p /models && chown fusion:fusion /models

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# The default agent: speech here, the LLM on the vLLM container beside this one
# (docker/docker-compose.gpu.yml). Mount your own over it: -v ./agent.py:/app/agent.py
COPY docker/agent.py /app/agent.py
USER fusion

EXPOSE 8000

# /health is the one endpoint that needs no key, precisely so this works.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# The agent is /app/agent.py (FUSION_AGENT): the default above, or yours mounted
# over it. 0.0.0.0 is refused unless keys are configured, which is the point.
ENTRYPOINT ["frun"]
CMD ["up", "--host", "0.0.0.0", "--port", "8000"]
