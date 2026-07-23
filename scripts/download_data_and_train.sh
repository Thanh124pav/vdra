#!/usr/bin/env bash
set -euo pipefail

# Download/cache HuggingFace math datasets, convert them to VERL parquet, then
# launch one training run. Override any setting with environment variables:
#
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#   ALGO=vdra N_GPUS_PER_NODE=4 TP_SIZE=2 TOTAL_STEPS=20 \
#   bash scripts/download_data_and_train.sh
#
# If your Docker image starts in the nested verl/ directory, run the wrapper:
#
#   bash scripts/download_data_and_train.sh
#
# Algorithms:
#   ALGO=vdra    tree rollout + VDRA budget allocation
#   ALGO=spo     tree rollout + uniform SPO tree
#   ALGO=treepo  tree rollout + TreePO-style ablation
#   ALGO=grpo    flat VERL GRPO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-deeplearning}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  else
    echo "[warn] CONDA_SH not found: ${CONDA_SH}; continuing without conda activate" >&2
  fi
fi

cd "${REPO_ROOT}"

# ------------------------- data preparation -------------------------
CONVERT_SOURCE="${CONVERT_SOURCE:-hf}"
INPUT_ROOT="${INPUT_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-verl/data}"
DATASETS="${DATASETS:-gsm8k math aime24 aime25 amc23 olympiadbench_hf}"
PROMPT_STYLE="${PROMPT_STYLE:-boxed}"
VERIFY_CONCAT="${VERIFY_CONCAT:-1}"
OVERWRITE_RAW="${OVERWRITE_RAW:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"

if [[ "${SKIP_DATA_PREP}" != "1" ]]; then
  convert_cmd=(
    python scripts/convert_hf_data_to_verl_parquet.py
    --source "${CONVERT_SOURCE}"
    --input-root "${INPUT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --prompt-style "${PROMPT_STYLE}"
    --exclude 'point24*'
    --datasets
  )
  # shellcheck disable=SC2206
  dataset_items=(${DATASETS})
  convert_cmd+=("${dataset_items[@]}")
  if [[ "${OVERWRITE_RAW}" == "1" ]]; then
    convert_cmd+=(--overwrite-raw)
  fi
  if [[ "${VERIFY_CONCAT}" == "1" ]]; then
    convert_cmd+=(--verify-concat)
  fi
  echo "[data] ${convert_cmd[*]}"
  "${convert_cmd[@]}"
else
  echo "[data] SKIP_DATA_PREP=1; using existing parquet under ${OUTPUT_ROOT}"
fi

# ------------------------- training config --------------------------
ALGO="${ALGO:-vdra}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TRAIN_DATASET="${TRAIN_DATASET:-math}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_DATASETS="${VAL_DATASETS:-math aime24 aime25 amc23}"
VAL_SPLIT="${VAL_SPLIT:-test}"

# Paths are relative to REPO_ROOT/verl because training runs from that directory.
TRAIN_FILE="${TRAIN_FILE:-data/${TRAIN_DATASET}/${TRAIN_SPLIT}.parquet}"
VAL_FILES="${VAL_FILES:-}"
if [[ -z "${VAL_FILES}" ]]; then
  VAL_FILES="["
  sep=""
  # shellcheck disable=SC2206
  val_items=(${VAL_DATASETS})
  for ds in "${val_items[@]}"; do
    VAL_FILES+="${sep}'data/${ds}/${VAL_SPLIT}.parquet'"
    sep=","
  done
  VAL_FILES+="]"
fi

PROJECT_NAME="${PROJECT_NAME:-vdra}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ALGO}-${TRAIN_DATASET}}"
LOGGER="${LOGGER:-'["console","wandb"]'}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
SHUFFLE="${SHUFFLE:-true}"
SEED="${SEED:-0}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-32}"
LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-32}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-600}"
TREE_SHAPE="${TREE_SHAPE:-'[6,6,6]'}"
ANSWER_PREFIX="${ANSWER_PREFIX:-null}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

COMMON_OVERRIDES=(
  "actor_rollout_ref.model.path=${MODEL}"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILES}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.shuffle=${SHUFFLE}"
  "+data.seed=${SEED}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.logger=${LOGGER}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_PER_GPU}"
)

if [[ -n "${VAL_BATCH_SIZE}" ]]; then
  COMMON_OVERRIDES+=("data.val_batch_size=${VAL_BATCH_SIZE}")
fi

case "${ALGO}" in
  vdra)
    ENTRYPOINT=(python -m recipe.gear_tree.main_gear_tree --config-path recipe/gear_tree/config --config-name gear_tree_trainer)
    MODE_OVERRIDES=(
      "gear_tree.tree_update_mode=spo"
      "gear_tree.gear.enabled=true"
      "gear_tree.gear.k_algorithm=budget_allocation"
      "gear_tree.gear.tail_mode=none"
      "gear_tree.tree_shape=${TREE_SHAPE}"
      "gear_tree.segment_length=${SEGMENT_LENGTH}"
      "gear_tree.answer_prefix=${ANSWER_PREFIX}"
    )
    ;;
  spo)
    ENTRYPOINT=(python -m recipe.gear_tree.main_gear_tree --config-path recipe/gear_tree/config --config-name gear_tree_trainer)
    MODE_OVERRIDES=(
      "gear_tree.tree_update_mode=spo"
      "gear_tree.gear.enabled=false"
      "gear_tree.vineppo_K=0"
      "gear_tree.tree_shape=${TREE_SHAPE}"
      "gear_tree.segment_length=${SEGMENT_LENGTH}"
      "gear_tree.answer_prefix=${ANSWER_PREFIX}"
    )
    ;;
  treepo|treepo_style_ablation)
    ENTRYPOINT=(python -m recipe.gear_tree.main_gear_tree --config-path recipe/gear_tree/config --config-name gear_tree_trainer)
    MODE_OVERRIDES=(
      "gear_tree.tree_update_mode=treepo_style_ablation"
      "gear_tree.gear.enabled=false"
      "gear_tree.tree_shape=${TREE_SHAPE}"
      "gear_tree.segment_length=${SEGMENT_LENGTH}"
      "gear_tree.answer_prefix=${ANSWER_PREFIX}"
    )
    ;;
  grpo)
    ENTRYPOINT=(python -m recipe.gear_tree.main_flat --config-path recipe/gear_tree/config --config-name flat_trainer)
    MODE_OVERRIDES=(
      "algorithm.adv_estimator=grpo"
      "reward_model.reward_manager=gear_math"
      "+gear_tree.answer_prefix=${ANSWER_PREFIX}"
    )
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use one of: vdra, spo, treepo, grpo" >&2
    exit 2
    ;;
esac

EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES_ARRAY=()
if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  # Space-separated Hydra overrides. For values with spaces, prefer editing this script.
  # shellcheck disable=SC2206
  EXTRA_OVERRIDES_ARRAY=(${EXTRA_OVERRIDES_STR})
fi

cd "${REPO_ROOT}/verl"
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

TRAIN_CMD=("${ENTRYPOINT[@]}" "${COMMON_OVERRIDES[@]}" "${MODE_OVERRIDES[@]}" "${EXTRA_OVERRIDES_ARRAY[@]}")

echo "[train] repo=${REPO_ROOT}"
echo "[train] algo=${ALGO} model=${MODEL} train=${TRAIN_FILE} val=${VAL_FILES}"
echo "[train] command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[train] SKIP_TRAIN=1; not launching training."
  exit 0
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[train] DRY_RUN=1; not launching training."
  exit 0
fi

exec "${TRAIN_CMD[@]}"
