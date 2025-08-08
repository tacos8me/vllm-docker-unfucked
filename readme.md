# vLLM Blackwell Docker Setup

Automated Docker Compose setup for running vLLM on NVIDIA Blackwell GPUs (RTX 5080/5090, B100/B200).

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA Container Runtime
- NVIDIA Driver R570+ (for CUDA 12.8+ support)
- NVIDIA Blackwell GPU (RTX 5080/5090, B100/B200)

### 1. Setup Files
Create these four files in your project directory:

```bash
# Create project directory
mkdir vllm-blackwell && cd vllm-blackwell

# Create the files (download from artifacts)
# - Dockerfile
# - docker-compose.yml
# - .env
# - README.md (this file)
```

### 2. Configure Environment
Edit `.env` file with your specific settings:

```bash
# Essential settings to modify:
MODELS_PATH=/your/models/path
MODEL_NAME=your-model-name
VLLM_PORT=8080
TENSOR_PARALLEL_SIZE=1  # Number of GPUs
```

### 3. Build and Start
```bash
# Build the image (takes 5-10 minutes)
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f vllm-server
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## ⚙️ Configuration Options

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

# Increase shared memory
SHM_SIZE=32g
```

### Performance Tuning
```bash
# High throughput
MAX_NUM_SEQS=512
GPU_MEMORY_UTIL=0.98

# Low latency
MAX_NUM_SEQS=64
GPU_MEMORY_UTIL=0.85
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

## 🐛 Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.9-base nvidia-smi

# Verify driver
nvidia-smi
```

**Memory Issues:**
```bash
# Reduce GPU memory usage
GPU_MEMORY_UTIL=0.8

# Increase shared memory
SHM_SIZE=32g
```

**Build Failures:**
```bash
# Clean build
docker-compose build --no-cache --pull

# Check CUDA availability during build
docker-compose exec vllm-server nvcc --version
```

**Flash Attention Errors:**
```bash
# Force Flash Attention 2
VLLM_FLASH_ATTN_VERSION=2
```

### Debugging Commands
```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs --details vllm-server

# Test PyTorch CUDA
docker-compose exec vllm-server python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute: {torch.cuda.get_device_capability(0)}')
"

# Test vLLM installation
docker-compose exec vllm-server python -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
"
```

## 🔧 Advanced Usage

### Custom Model Loading
```python
# Python client example
import requests

response = requests.post('http://localhost:8080/v1/chat/completions', 
    json={
        "model": "your-model",
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "max_tokens": 200,
        "temperature": 0.7
    })

print(response.json())
```

### Batch Processing
```bash
# Use vLLM CLI for batch inference
docker-compose exec vllm-server python -m vllm.entrypoints.llm \
  --model /workspace/models/your-model \
  --input-file /workspace/inputs.txt \
  --output-file /workspace/outputs.txt
```

### Load Balancing
For production, use multiple instances:

```yaml
# docker-compose.override.yml
services:
  vllm-server-1:
    extends: vllm-server
    container_name: vllm-server-1
    ports: ["8081:8080"]
    
  vllm-server-2:
    extends: vllm-server
    container_name: vllm-server-2
    ports: ["8082:8080"]
```

## 📈 Performance Tips

1. **Memory Optimization:**
   - Use `bfloat16` for Blackwell GPUs
   - Set `GPU_MEMORY_UTIL=0.95` for maximum utilization

2. **Multi-GPU:**
   - Set `TENSOR_PARALLEL_SIZE` to number of GPUs
   - Use `ipc: host` for fast inter-GPU communication

3. **Batch Processing:**
   - Increase `MAX_NUM_SEQS` for higher throughput
   - Adjust `MAX_MODEL_LEN` based on your use case

4. **Blackwell Optimization:**
   - Ensure `TORCH_CUDA_ARCH_LIST` includes `12.0+PTX`
   - Use `VLLM_FLASH_ATTN_VERSION=2` for stability

## 📝 Example Use Cases

### Chat Service
```bash
MODEL_NAME=microsoft/DialoGPT-medium
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=128
```

### Code Generation
```bash
MODEL_NAME=codellama/CodeLlama-7b-Python-hf
MAX_MODEL_LEN=4096
TEMPERATURE=0.1
```

### Large Models
```bash
MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
TENSOR_PARALLEL_SIZE=4
SHM_SIZE=64g
MEM_LIMIT=256g
```

---

🎉 **You're ready to run vLLM on Blackwell GPUs!** 

For more advanced configuration, see the [vLLM documentation](https://docs.vllm.ai/).