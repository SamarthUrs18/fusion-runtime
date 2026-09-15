# fusion-runtime Dockerfile
# Multi-stage build for minimal production image

# ============ Build Stage ============
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install llama.cpp with CUDA support
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Install ONNX Runtime with CUDA
RUN pip install onnxruntime-gpu

# Install faster-whisper
RUN pip install faster-whisper

# Install other dependencies
RUN pip install \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pydantic-settings \
    numpy \
    scipy \
    soundfile \
    websockets \
    aiohttp \
    python-multipart

# ============ Runtime Stage ============
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser fusion_runtime/ ./fusion_runtime/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create model cache directory
RUN mkdir -p /models && chown appuser:appuser /models

# Switch to app user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()"

# Run server
CMD ["python", "-m", "fusion_runtime.server"]