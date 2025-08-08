# =============================================================================
# Dockerfile for vLLM with Blackwell GPU Support
# =============================================================================
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Prevent interactive dialogs during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Build optimization settings
ENV MAX_JOBS=4
ENV NVCC_THREADS=2
ENV PIP_NO_CACHE_DIR=1

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

# Install PyTorch with CUDA 12.9 support
RUN /root/.local/bin/uv pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Clone vLLM repository
WORKDIR /opt
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /opt/vllm

# Build vLLM from source
RUN python use_existing_torch.py && \
    /root/.local/bin/uv pip install -r requirements/build.txt && \
    /root/.local/bin/uv pip install -e . --no-build-isolation

# Create models directory
RUN mkdir -p /workspace/models

# Verify installation
RUN python -c "import vllm, torch; print(f'✅ vLLM {vllm.__version__} with PyTorch {torch.__version__} installed successfully')"

# Set working directory
WORKDIR /opt/vllm

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import vllm; import torch; assert torch.cuda.is_available()" || exit 1

# Default command - can be overridden in docker-compose
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--model", "/workspace/models"]