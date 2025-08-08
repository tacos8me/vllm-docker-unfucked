# =============================================================================
# vLLM Blackwell Docker Configuration
# =============================================================================

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Path to your models directory on the host
MODELS_PATH=/models

# Model name to serve (relative to MODELS_PATH or HuggingFace model ID)
MODEL_NAME=microsoft/DialoGPT-medium
# Examples:
# MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
# MODEL_NAME=microsoft/DialoGPT-medium
# MODEL_NAME=/workspace/models/your-local-model

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
# Number of GPUs to use for tensor parallelism
TENSOR_PARALLEL_SIZE=1

# Maximum model sequence length
MAX_MODEL_LEN=4096

# Maximum number of sequences to process in parallel
MAX_NUM_SEQS=256

# GPU memory utilization (0.0 to 1.0)
GPU_MEMORY_UTIL=0.95

# Data type for model weights (auto, float16, bfloat16)
DTYPE=auto

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
# Port to expose the API server
VLLM_PORT=8080

# Shared memory size for multi-GPU communication
SHM_SIZE=16g

# Memory limit for container
MEM_LIMIT=64g

# CUDA devices to use (all, or specific like "0,1")
CUDA_VISIBLE_DEVICES=all

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
# Cache directory for HuggingFace models
CACHE_PATH=./cache

# Logs directory
LOGS_PATH=./logs

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Logging level (DEBUG, INFO, WARNING, ERROR)
VLLM_LOGGING_LEVEL=INFO

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================
# Additional arguments to pass to vLLM server
ADDITIONAL_ARGS=

# Examples of additional arguments:
# ADDITIONAL_ARGS=--enable-lora --max-lora-rank 16
# ADDITIONAL_ARGS=--quantization awq
# ADDITIONAL_ARGS=--enable-prefix-caching

# =============================================================================
# OPTIONAL: MONITORING CONFIGURATION
# =============================================================================
# Grafana admin password (if using monitoring)
GRAFANA_PASSWORD=admin123