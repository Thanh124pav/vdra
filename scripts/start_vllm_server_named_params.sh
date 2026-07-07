#!/bin/bash

# Default values for parameters
GPU_IDX=0
GPU_MEM_UTILIZATION=0.9
MAX_NUM_SEQS=256
ENABLE_PREFIX_CACHING=false
DISABLE_SLIDING_WINDOW=false
DISABLE_FRONTEND_MULTIPROCESSING=false
MAX_MODEL_LEN=""
DTYPE="${VLLM_DTYPE:-bfloat16}"

# Parse named parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --swap-space) SWAP_SPACE="$2"; shift ;;
        --gpu-idx) GPU_IDX="$2"; shift ;;
        --gpu-memory-utilization) GPU_MEM_UTILIZATION="$2"; shift ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift ;;
        --enable-prefix-caching) ENABLE_PREFIX_CACHING=true ;;
        --disable-sliding-window) DISABLE_SLIDING_WINDOW=true ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        --disable-frontend-multiprocessing) DISABLE_FRONTEND_MULTIPROCESSING=true ;;
        --dtype) DTYPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export VLLM_HF_FOLDER_CACHE_FILE=$HF_HOME/vllm_hf_folder_cache.json

ARGS=(
    --model "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --seed "$SEED"
    --gpu-memory-utilization "$GPU_MEM_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
)

if [ -z "$DTYPE" ]; then
    DTYPE=bfloat16
fi
ARGS+=(--dtype "$DTYPE")

if [ -n "${SWAP_SPACE:-}" ]; then
    ARGS+=(--swap-space "$SWAP_SPACE")
fi
if [ "$ENABLE_PREFIX_CACHING" = true ]; then
    ARGS+=(--enable-prefix-caching)
fi
if [ "$DISABLE_SLIDING_WINDOW" = true ]; then
    ARGS+=(--disable-sliding-window)
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ "$DISABLE_FRONTEND_MULTIPROCESSING" = true ]; then
    ARGS+=(--disable-frontend-multiprocessing)
fi

CUDA_VISIBLE_DEVICES=$GPU_IDX python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
