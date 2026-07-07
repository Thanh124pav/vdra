#!/usr/bin/env bash
# Evaluate a trained GEAR/SPO checkpoint with the same lighteval recipe SPO
# uses (math_500, gsm8k). Wrapper around SPO's scripts/evaluate_long_cot.sh
# so we can run with one invocation.
#
# Usage:
#   bash scripts/evaluate_long_cot.sh <hf_or_local_model_path> [task=math_500]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:?Usage: evaluate_long_cot.sh <model> [task]}"
TASK="${2:-math_500}"
NUM_GPUS="${NUM_GPUS:-1}"
CTX="${CTX:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${GEAR_ROOT}/data/evals/${MODEL//[\/]/__}}"
mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16"
MODEL_ARGS+=",max_model_length=${CTX}"
MODEL_ARGS+=",gpu_memory_utilization=0.8"
MODEL_ARGS+=",data_parallel_size=${NUM_GPUS}"
MODEL_ARGS+=",generation_parameters={max_new_tokens:${CTX},temperature:0.6,top_p:0.95}"

lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
  --use-chat-template \
  --output-dir "${OUTPUT_DIR}"
