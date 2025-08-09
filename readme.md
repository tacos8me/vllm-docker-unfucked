# vLLM Docker Unfucked 🌮

**Because setting up vLLM shouldn't make you want to throw your GPU out the window.**

This repo provides a working Docker Compose setup for vLLM that actually works with modern NVIDIA GPUs, especially Blackwell (RTX 5080/5090, B100/B200). No more CUDA version hell, no more mysterious build failures, no more "why the fuck doesn't this work" moments.

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

## 🚀 Quick Start (The Actually Quick Way)

### Prerequisites
- Docker with NVIDIA Container Runtime
- NVIDIA Driver R570+ (for CUDA 12.8+ support)  
- Any modern NVIDIA GPU (optimized for Blackwell but works with others)
- Basic sanity (optional)

### 1. Clone This Repo
```bash
git clone https://github.com/tacos8me/vllm-docker-unfucked.git
cd vllm-docker-unfucked
```

### 2. Configure Your Shit
Edit `.env` file with your actual paths (don't just copy-paste like a noob):

```bash
# CHANGE THESE OR IT WON'T WORK:
MODELS_PATH=/your/actual/models/path  # Where your models live
MODEL_NAME=glm-air                   # Local model name or HF model ID
SERVED_MODEL_NAME=oai/glm            # OpenAI-compatible model name
VLLM_PORT=8080                       # Port to expose
TENSOR_PARALLEL_SIZE=1               # Number of GPUs (1 unless you're rich)
MAX_MODEL_LEN=65536                  # Context length (65K for long conversations)
MAX_NUM_SEQS=2048                    # Parallel sequences (adjust for your GPU)
GPU_MEMORY_UTIL=0.95                 # Use 95% of GPU memory
```

### 3. Build and Start (The Moment of Truth)
```bash
# Build the image (takes a LONG TIME, find something else to do)
docker-compose build

# Start the magic
docker-compose up -d

# Watch it work (or fail spectacularly)
docker-compose logs -f vllm-server
```

### 4. Test That It Actually Works
```bash
# Health check
curl http://localhost:8080/health

# Chat completion (use your SERVED_MODEL_NAME)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oai/glm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Check monitoring dashboards
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:9090  # Prometheus
open http://localhost:8081/metrics  # vLLM metrics
```

## 🔥 What This Unfucks

### The Usual vLLM Docker Pain Points:
- ❌ **"CUDA error: no kernel image available"** - Because someone thought sm_120 wasn't important
- ❌ **Flash Attention version hell** - FA3 doesn't work with Blackwell, but who reads docs?
- ❌ **PyTorch/CUDA version mismatches** - It's like dependency hell but with more crying
- ❌ **Manual container management** - Typing the same docker run command 47 times
- ❌ **"Works on my machine"** - But your machine is from 2019

### What We Actually Fixed:
- ✅ **Proper Blackwell support** - sm_120 compute capability baked in
- ✅ **Flash Attention 2** - Because FA3 is still broken for Blackwell
- ✅ **CUDA 12.9 + PyTorch compatibility** - They actually talk to each other
- ✅ **Automated everything** - Docker Compose handles the bullshit
- ✅ **Multi-GPU ready** - Just change one number in .env
- ✅ **Production ready** - Health checks, logging, monitoring stack included
- ✅ **Advanced performance tuning** - Prefix caching, chunked prefill, optimized batching
- ✅ **Full monitoring stack** - Prometheus + Grafana dashboards out of the box
- ✅ **Configurable build settings** - Tune compilation for your hardware

## 💻 Supported Hardware

### Definitely Works:
- **NVIDIA RTX 5090/6000 Pro** (Blackwell) - This is what we optimized for
- **NVIDIA B100/B200** (Blackwell Data Center) - You should have this figured out already. But yes
- **NVIDIA RTX 4090/4080** (Ada Lovelace) - Yut
- **NVIDIA H100/H200** (Hopper) - Sure

### Probably Works:
- Most RTX 30 series and above
- Tesla/Quadro cards with compute capability 7.0+
- Anything that doesn't make your electricity bill cry

## ⚙️ Configuration Options (The Good Stuff)

### Model Configuration
```bash
# Local model
MODEL_NAME=/workspace/models/llama-2-7b-chat

# HuggingFace model (auto-downloaded)
MODEL_NAME=microsoft/DialoGPT-medium
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```

### Multi-GPU Setup
```bash
# Use 2 GPUs
TENSOR_PARALLEL_SIZE=2
CUDA_VISIBLE_DEVICES=0,1

# Increase shared memory and container limits
SHM_SIZE=32g
MEM_LIMIT=256g

# Build optimization for multi-GPU
MAX_JOBS=32
NVCC_THREADS=16
```

### Performance Tuning
```bash
# High throughput setup
MAX_NUM_SEQS=2048
MAX_MODEL_LEN=65536
GPU_MEMORY_UTIL=0.95
ENABLE_PREFIX_CACHING=true
ADDITIONAL_ARGS=--enable-chunked-prefill --max-num-batched-tokens 8192

# Low latency setup
MAX_NUM_SEQS=256
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.90
BLOCK_SIZE=16

# Memory optimization
KV_CACHE_DTYPE=auto
QUANTIZATION=none
SWAP_SPACE=4
```

## 🛠️ Management Commands

### Basic Operations
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart with new config
docker-compose down && docker-compose up -d

# View logs
docker-compose logs -f vllm-server

# Execute commands in container
docker-compose exec vllm-server bash
```

### Updating
```bash
# Rebuild image with latest vLLM
docker-compose build --no-cache

# Pull latest base image
docker-compose pull
```

### Cleanup
```bash
# Remove containers and networks
docker-compose down

# Remove images and volumes
docker-compose down --rmi all --volumes

# Full cleanup
docker system prune -a
```

## 📊 Monitoring (Optional)

Enable Prometheus and Grafana by uncommenting the monitoring services in `docker-compose.yml`:

```bash
# Start with monitoring
docker-compose up -d

# Access dashboards
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana (admin/admin123)
```

## 🐛 Troubleshooting (When Shit Breaks)

### "GPU Not Detected" (Classic)
```bash
# Check if NVIDIA runtime is actually working
docker run --rm --gpus all nvidia/cuda:12.9-base nvidia-smi

# If that fails, your Docker setup is fucked
# Fix: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### "Out of Memory" (Buy More VRAM)
```bash
# Reduce GPU memory usage (in .env)
GPU_MEMORY_UTIL=0.8

# Or increase shared memory if multi-GPU
SHM_SIZE=32g
```

### "Build Failed" (Murphy's Law)
```bash
# Nuclear option - clean everything
docker-compose down --rmi all --volumes
docker system prune -a

# Try again with more logging
docker-compose build --no-cache --progress=plain
```

### "Flash Attention Errors" (Because Nothing Is Easy)
```bash
# Force Flash Attention 2 (add to .env)
VLLM_FLASH_ATTN_VERSION=2

# Or disable Flash Attention entirely (slower but works)
ADDITIONAL_ARGS=--disable-flash-attn
```

### Debugging Commands (For When You're Really Stuck)
```bash
# Check if the container is actually running
docker-compose ps

# See what the hell is happening
docker-compose logs --follow --tail=100 vllm-server

# Get inside the container (for advanced debugging)
docker-compose exec vllm-server bash

# Test PyTorch CUDA inside container
docker-compose exec vllm-server python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"None\"}')
"
```

## 🚀 Production Tips (For The Pros)

### Memory Optimization
```bash
# Use bfloat16 for Blackwell GPUs (faster + less memory)
DTYPE=bfloat16

# Max out GPU memory utilization
GPU_MEMORY_UTIL=0.98
```

### Multi-GPU Setup (Show Off Mode)
```bash
# Use all your GPUs
TENSOR_PARALLEL_SIZE=4
CUDA_VISIBLE_DEVICES=0,1,2,3

# Increase shared memory for multi-GPU communication
SHM_SIZE=64g
```

### High Throughput Setup
```bash
# Maximum parallel sequences for throughput
MAX_NUM_SEQS=2048

# Long context for complex tasks
MAX_MODEL_LEN=65536

# Advanced performance features
ENABLE_PREFIX_CACHING=true
ADDITIONAL_ARGS=--enable-chunked-prefill --max-num-batched-tokens 8192 --enable-prefix-caching

# Build optimization
MAX_JOBS=32
NVCC_THREADS=16

# Memory optimization
KV_CACHE_DTYPE=auto
SWAP_SPACE=4
MAX_PARALLEL_LOADING_WORKERS=4
```

## 📊 Monitoring Stack (Enabled by Default)

We've included a full monitoring stack with Prometheus and Grafana:

```bash
# Start everything (monitoring included)
docker-compose up -d

# Access the monitoring stack
open http://localhost:3000  # Grafana dashboards (admin/admin123)
open http://localhost:9090  # Prometheus metrics
open http://localhost:8081/metrics  # Raw vLLM metrics
```

### What You Get:
- **Real-time performance metrics** - Request latency, throughput, GPU utilization
- **Memory usage tracking** - KV cache, model weights, system memory
- **Request analytics** - Success rates, error tracking, queue depths
- **Hardware monitoring** - GPU temperature, power consumption, CUDA errors
- **30-day data retention** - Historical performance analysis

### Monitoring Configuration:
```bash
# Disable monitoring (if you're boring)
ENABLE_MONITORING=false

# Custom ports
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090
METRICS_PORT=8081

# Grafana admin password
GRAFANA_PASSWORD=admin123
```

## 🤝 Contributing

Found a bug? Setup doesn't work? Have a better way to do something?

1. **Open an issue** - Tell us what's broken
2. **Submit a PR** - Fix it yourself (we love you)
3. **Share feedback** - What worked, what didn't

## 📝 Example Model Configs

### For Chat/Conversation (GLM-4.5-Air)
```bash
MODEL_NAME=glm-air
SERVED_MODEL_NAME=oai/glm
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=1024
DTYPE=auto
ENABLE_PREFIX_CACHING=true
ADDITIONAL_ARGS=--enable-chunked-prefill --max-num-batched-tokens 8192
```

### For Code Generation  
```bash
MODEL_NAME=codellama/CodeLlama-7b-Python-hf
MAX_MODEL_LEN=4096
TEMPERATURE=0.1
DTYPE=bfloat16
```

### For Large Models (RTX 4090+ Recommended)
```bash
MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
TENSOR_PARALLEL_SIZE=2
SHM_SIZE=32g
MEM_LIMIT=256g
DTYPE=auto
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=512
GPU_MEMORY_UTIL=0.95
QUANTIZATION=none
ENABLE_PREFIX_CACHING=true
```

## ⚠️ Known Issues

- **Flash Attention 3**: Doesn't work with Blackwell yet (use FA2)
- **Some quantized models**: May need specific ADDITIONAL_ARGS
- **Very old GPUs**: Compute capability < 7.0 not supported
- **WSL2**: Works but can be flaky with GPU passthrough

## 🎯 Why This Exists

Because the official vLLM Docker setup is a maze of conflicting documentation, version incompatibilities, and "works on the maintainer's machine" syndrome. This repo provides a working setup that actual humans can use without a PhD in DevOps.

**TL;DR**: We did the painful setup work so you don't have to. You're welcome.

---

## 📜 License

MIT License - Do whatever you want with this code. If it breaks your system, that's on you. 🤷‍♂️

---

**Made with ☕ and spite by people tired of Docker setup hell.**

🌮 **Star this repo if it saved you from throwing your computer out the window.**