# =============================================================================
# Dockerfile for vLLM with Blackwell GPU Support
# =============================================================================
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Prevent interactive dialogs during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Build optimization settings (configurable via build args)
ARG MAX_JOBS=32
ARG NVCC_THREADS=16
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
# Allow pip/uv to leverage cache between layers for faster rebuilds
ENV PIP_NO_CACHE_DIR=0

# CUDA environment for Blackwell
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# CRITICAL: Blackwell compute capabilities + PTX for forward compatibility
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX"

# Flash Attention 2 for Blackwell compatibility
ENV VLLM_FLASH_ATTN_VERSION=2

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ca-certificates \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create and activate virtual environment
RUN /root/.local/bin/uv venv /opt/vllm-env --python 3.12
ENV VIRTUAL_ENV=/opt/vllm-env
ENV PATH="/opt/vllm-env/bin:${PATH}"

# Install PyTorch (optionally pinned via build args; if unset, latest matching cu index is used)
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN bash -lc 'set -euo pipefail; \
  TORCH_SPEC=${TORCH_VERSION:+torch==${TORCH_VERSION}}; TORCH_SPEC=${TORCH_SPEC:-torch}; \
  VISION_SPEC=${TORCHVISION_VERSION:+torchvision==${TORCHVISION_VERSION}}; VISION_SPEC=${VISION_SPEC:-torchvision}; \
  AUDIO_SPEC=${TORCHAUDIO_VERSION:+torchaudio==${TORCHAUDIO_VERSION}}; AUDIO_SPEC=${AUDIO_SPEC:-torchaudio}; \
  /root/.local/bin/uv pip install "$TORCH_SPEC" "$VISION_SPEC" "$AUDIO_SPEC" --index-url ${TORCH_CUDA_INDEX_URL}'

# Clone vLLM repository (shallow). If VLLM_REF is provided, checkout that ref; otherwise clone default branch
ARG VLLM_REF
WORKDIR /opt
RUN bash -lc 'set -euo pipefail; \
  if [ -n "${VLLM_REF:-}" ]; then \
    git clone --depth 1 --branch "${VLLM_REF}" https://github.com/vllm-project/vllm.git; \
  else \
    git clone --depth 1 https://github.com/vllm-project/vllm.git; \
  fi'
WORKDIR /opt/vllm

# Build vLLM from source (non-editable install for faster cold start and smaller runtime deps)
RUN python use_existing_torch.py && \
    /root/.local/bin/uv pip install -r requirements/build.txt && \
    /root/.local/bin/uv pip install . --no-build-isolation

# Create models directory
RUN mkdir -p /workspace/models

# Verify installation
RUN python -c "import vllm, torch; print(f'✅ vLLM {vllm.__version__} with PyTorch {torch.__version__} installed successfully')"

# Set working directory
WORKDIR /opt/vllm

# Expose API and metrics ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import vllm; import torch; assert torch.cuda.is_available()" || exit 1

# Create non-root user and set permissions for mounted paths
RUN useradd -m -u 10001 -s /bin/bash vllm && \
    mkdir -p /workspace/models /opt/vllm/logs && \
    chown -R vllm:vllm /workspace /opt/vllm /root/.cache || true

# Default command - overridden by docker-compose
USER vllm