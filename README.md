# treetune: Unified RL Framework for Reasoning LLMs

Một codebase thống nhất để huấn luyện LLM trên các tác vụ suy luận (MATH, GSM8K, Point24) bằng nhiều thuật toán RL khác nhau. Tất cả các thuật toán đứng ngang hàng, chọn bằng config.

## Algorithm catalog

| Thuật toán | Trainer | Episode generator | Inference strategy | Script |
|------------|---------|-------------------|--------------------|--------|
| **PPO** — vanilla Proximal Policy Optimization | `ppo` | `math_episode_generator` | `cot` | `train_ppo_MATH.sh` |
| **GRPO** — Group Relative PO (DeepSeek) | `ppo` | `math_episode_generator_w_group_advantages` (adv=`grpo`) | `cot` | `train_grpo_MATH.sh` |
| **RLOO** — REINFORCE Leave-One-Out | `ppo` | `math_episode_generator_w_group_advantages` (adv=`rloo`) | `cot` | `train_rloo_GSM8K.sh` |
| **VinePPO** — PPO với vine-style value baseline | `ppo` | `vineppo_episode_generator` | `cot` | `train_vineppo_GSM8K.sh` |
| **DPO** — Direct Preference Optimization (positive variant) | `dpo_positive` | `math_dpo_positive_episode_generator` | `cot` | `train_dpo_MATH.sh` |
| **RestEM** — Rejection sampling + EM-style FT | `restem` | `math_restem_episode_generator` | `cot` | `train_restem_MATH.sh` |
| **SPO-chain** — Segment PO trên chain | `ppo` | `math_episode_generator` | `cot` | `train_spo_chain_MATH.sh` |
| **SPO-tree** — Segment PO trên cây branching | `ppo` | `hybrid_episode_generator` | `hybrid` | `train_spo_tree_MATH.sh` |
| **GEAR** — Information-Gated PO (SPO + ValueShare + Prune) | `ppo` | `gear_episode_generator` | `gear` | `train_gear_tree_MATH.sh` |
| **GEAR-SPO-chain** — GEAR prune/allocate on chain-style rollout | `ppo` | `gear_episode_generator` | `gear` | `train_gear_spo_chain_MATH.sh` |
| **GEAR-VinePPO** — VinePPO advantages with GEAR-controlled rollout | `ppo` | `gear_vineppo_episode_generator` | `gear` | `train_gear_vineppo_MATH.sh` |

Mỗi thuật toán có file canonical ở `configs/algorithms/<algo>.libsonnet` — thin overlay set `(trainer, episode_generator, inference_strategy)` types. Người dùng compose với model/task base để tạo full experiment config.

## Cấu trúc thư mục

```
gear/
├── treetune/                         # Python package thống nhất
│   ├── common/                       # registry, FromParams, Params utilities
│   ├── trainers/                     # ppo, dpo_positive, restem, mle, ...
│   ├── episode_generators/           # tất cả episode generators (PPO/GRPO/DPO/RestEM/VinePPO/SPO/GEAR)
│   ├── inference_strategies/         # cot, hybrid, gear, ...
│   ├── gear/                        # GEAR core helpers (TV bound, budget allocation, local gates)
│   ├── runtime/                      # policy iteration runtime
│   ├── models/, tasks/, analyzers/   # SPO infrastructure
│   └── main.py                       # entry point (treetune.main)
├── guidance/                         # vendored guidance lib (parsing prompts)
├── configs/
│   ├── algorithms/                   # ppo.libsonnet, grpo.libsonnet, ... (9 files)
│   ├── trainers/, episode_generators/, inference_strategies/, models/, tasks/
│   ├── polIter_<model>_<algo>_<dataset>.jsonnet  # full experiment configs
│   ├── ablations/, baselines/        # GEAR-specific overlays
│   ├── gear_defaults.libsonnet, gear_overlay.libsonnet
│   └── episode_generators/branch_factor_*.jsonnet  # tree shape overlays
├── scripts/                          # train_<algo>_<dataset>.sh + utilities
├── tests/                            # unit tests
├── docs/legacy/                      # legacy SPO README/LICENSE/Dockerfile
└── README.md
```

## Bắt đầu nhanh

### Cài đặt

```bash
bash scripts/setup.sh
```

### Khởi động vLLM server (cần cho scoring)

```bash
bash scripts/start_vllm_server.sh /path/to/model 8000 42 32 0
export APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1
```

### Huấn luyện một thuật toán

```bash
# PPO trên MATH với model mặc định
bash scripts/train_ppo_MATH.sh

# GRPO với model khác
MODEL=deepseekR1Qwen bash scripts/train_grpo_MATH.sh

# DPO
bash scripts/train_dpo_MATH.sh

# VinePPO trên GSM8K
bash scripts/train_vineppo_GSM8K.sh

# SPO-tree với tree shape tùy chỉnh
TREE=6666 bash scripts/train_spo_tree_MATH.sh

# GEAR-tree (mặc định 666)
GEAR_TREE=666 bash scripts/train_gear_tree_MATH.sh
```

### Đánh giá

```bash
bash scripts/evaluate.sh polIter_qwen1_5b_base_gear_tree_MATH \
    experiments/gear-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained

# Chỉ đánh giá AIME 2024
bash scripts/evaluate.sh polIter_qwen1_5b_base_gear_tree_MATH \
    experiments/gear-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --dataset aime24

# Đánh giá nhiều dataset
bash scripts/evaluate.sh polIter_qwen1_5b_base_gear_tree_MATH \
    experiments/gear-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --dataset math aime24 olympiadbench \
    --debug_mode=true

# Ghép thêm config override; config sau override config trước
bash scripts/evaluate.sh polIter_qwen1_5b_base_gear_tree_MATH \
    experiments/gear-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --config local/math_local_10 \
    --config local/math_local_runtime \
    --dataset math

# Cũng có thể truyền danh sách config phân tách bằng dấu phẩy
bash scripts/evaluate.sh \
    polIter_qwen1_5b_base_gear_tree_MATH,local/math_local_10 \
    experiments/gear-tree-666-qwen1.5b-math/iteration_0010/hf_pretrained \
    --dataset math

# Đổi model/checkpoint, tokenizer, context và generation limit trực tiếp
bash scripts/evaluate.sh \
    polIter_deepseekR1Qwen_gear_tree_MATH \
    HuggingFaceTB/SmolLM2-135M \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --context-length 4096 \
    --max-new-tokens 1024 \
    --dataset aime24

# Eval lần lượt mọi checkpoint trong một experiment
APP_EXPERIMENT_NAME=eval-training-sweep \
bash scripts/evaluate.sh \
    polIter_deepseekR1Qwen_gear_tree_MATH \
    experiments/gear-tree-666-deepseekR1Qwen-math \
    --all-checkpoints \
    --context-length 4096 \
    --datasets aime24,aime25

# Xem danh sách alias
bash scripts/evaluate.sh --list-datasets
```

Nếu không truyền `--dataset` hoặc `--datasets`, script vẫn chạy toàn bộ
`inference_pipelines` trong config như trước.

Sau mỗi dataset, metric được in lên terminal và append vào:

```text
experiments/<eval-name>/evaluation/iteration__0/evaluation_results.jsonl
```

## Compose configs

Mỗi config experiment ghép từ các lớp overlay:

```
[gvar.jsonnet]                              # global vars
+ [prompt_library/<task>.jsonnet]           # task-specific prompts
+ [runtimes/policy_iteration.jsonnet]       # runtime
+ [episode_generators/<eg>.jsonnet]         # episode generator type
+ [trainers/<algo>_<dataset>.jsonnet]       # trainer hyper-params
+ [models/<model>.jsonnet]                  # model
+ {custom overrides}
```

Để tạo experiment mới, copy một `polIter_*.jsonnet` đã có rồi đổi các overlay.

Mỗi model/config vẫn giữ context mặc định riêng. Có thể override đồng bộ cho
mọi model và giải thuật bằng biến môi trường `APP_MAX_MODEL_LEN`. Override được
áp dụng sau khi merge Jsonnet cho các giới hạn sequence/trainer, context của
node expander và tất cả vLLM server (train, evaluation, analyzer):

```bash
APP_MAX_MODEL_LEN=4096 bash scripts/train_gear_tree_deepseekR1Qwen_MATH.sh
```

## VDRA - pruning and adaptive rollout allocation

The default adaptive path is VDRA, not the legacy perplexity predictor. Each
node produces two separate signals:

- `predicted_k`: useful branch demand and pruning cap;
- `vdra_dispersion_C`: the upper bound \(C_s\) used to prioritize residual budget.

Pilot handling (updated):

- Pilots that emit EOS inside the first phase are **shortcut** children: they
  are complete trajectories, excluded from TV estimation, graded immediately
  and counted against the node's branch budget (`vdra_pilot_children_shortcut`,
  `vdra_shortcut_overage`).
- Duplicate pruning removes the pilot with the **most duplicate partners**
  (pairwise TV below the threshold) until no duplicate pair remains.
- Reuse selection among the surviving pilots is a **seeded uniform draw** —
  never likelihood-ranked, so the reused children stay an unbiased sample of
  the continuation distribution as far as pruning permits.

After pruning, saved branches enter a shared pool. Nodes with remaining demand
receive additional branches through capped water filling for
\(\min\sum_s C_s/k_s\). The continuous priority is `sqrt(C_s)` and integer
allocations use the exact bounded marginal integer solver by default (`rounding_strategy=integer_marginal`); relax-and-round modes remain ablations only.
`budget_lambda` is not a VDRA parameter; the solver's internal `dual_lambda`
is computed, never tuned. In the verl online runtime sibling frontier nodes
are expanded concurrently so allocation queues genuinely batch (check
`queue_size_at_flush` in `queue_flushes.jsonl`).

Main configuration:

| Parameter | Default | Meaning |
|---|---:|---|
| `pilot_branch_factor` | branch factor | Pilot children \(k_0\); must exceed the max default branch factor for residual redistribution |
| `likelihood_samples_per_distribution` | 2 | Mixture samples \(r\) per pilot child |
| `tv_second_phase_tokens` | 60 | Short horizon \(m\) |
| `n_min` | 1 | Minimum retained branch demand |
| `strict_vdra` | true | Fail rather than changing estimator/allocation behavior |
| `invalid_support_policy` | `error` | Handling for missing pair-specific support |
| `budget_mode` | `fixed_main` | `fixed_main` or `fixed_total_generated` |
| `allocation_proxy` | `vdra` | `vdra` / `uniform` / `random` / `direct_tv` / `empirical_variance` / `external_score` / `oracle` |
| `oracle_rollouts_per_node` | 16 | Full rollouts per node for the oracle proxy (eval-only) |
| `rounding_strategy` | `integer_marginal` | `integer_marginal`; legacy ablations: `largest_remainder` / `nearest_repair` / `stochastic` |
| `queue_capacity` | 8 | Nodes per online allocation flush |
| `root_allocation` | backend-specific | Joint root allocation; unsupported backends reject it |

`fixed_main` keeps the main-expansion budget fixed and reports pilot/scoring as
overhead. `fixed_total_generated` (verl online runtime) places pilot, support
and main generation under one generated-token cap equal to the matched uniform-SPO
expected token count; cap accounting lands in `vdra_token_cap`,
`vdra_generated_tokens_under_cap` and `vdra_token_cap_hit_count`. Neither mode
hides likelihood-scoring work.

The `empirical_variance` and `oracle` proxies generate and grade full rollouts
per node (cost in `vdra_proxy_rollout_tokens`); oracle runs are flagged
`run_valid_for_main_results: false` in the manifest. `external_score` takes an
import path via `external_score_module` (`module:callable`).

Strict main runs require a compatible tail-calibration artifact:

```bash
python scripts/calibrate_tail_divergence.py \
  --api-base http://127.0.0.1:8000/v1 --model "$MODEL" \
  --prompts-file data/math/train.jsonl --dataset math \
  --k0 6 --r 2 --horizons 60 --full-tokens 512 --grade

EPS_TAIL_CALIBRATION_PATH=artifacts/tail_calibration/<artifact>.json \
  bash scripts/train_gear_tree_MATH.sh
```

The calibration grader evaluates `pilot_response + continuation`. Artifacts
record model/checkpoint, dataset, \(k_0\), \(r\), horizons, seed and quantile.

Legacy `simple`, `perplexity`, `legacy_abs`, `simulation_lemma`, no-tail,
no-floor and no-queue paths are explicit ablations only. Presets live in
`configs/ablations/` and are launched through `scripts/run_ablations.sh`.

Offline experiment scripts:

- `scripts/calibrate_tail_divergence.py` — RQ2/RQ3/RQ4 + Direction A/B/D
  (tail quantiles, adaptive lookahead report, oracle dispersion, allocation
  regret);
- `scripts/eval_value_mse.py` — RQ5: MSE of \(\hat V(s)\) vs a high-budget
  reference under each allocation method;
- `scripts/eval_gradient_quality.py` — RQ6: cos/L2/variance of the segment
  gradient vs a high-budget reference gradient (small HF model, offline).

## Logging offline (no internet)

GEAR ghi mọi metric ra file để dùng offline:

- `<exp>/training_timing.jsonl` — mỗi iteration 1 dòng: `train_total_seconds` (không gồm eval), `eval_seconds`, cumulative wall.
- `<exp>/gear_demos/demos.jsonl` — mỗi tree: stats, per_depth, tree_construction_seconds, budget/local-gate demos.
- `<exp>/gear_demos/demos.md` — bản Markdown human-readable.

Xem live:

```bash
bash scripts/tail_demos.sh <exp_name>             # markdown
bash scripts/tail_demos.sh <exp_name> jsonl       # jsonl
python scripts/inspect_demos.py <exp>/gear_demos/demos.jsonl --summary
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q       # unit tests (no GPU needed)
bash scripts/run_smoke.sh                     # config compile + unit tests
bash scripts/train_debug.sh                   # E2E 2 iterations, depth 2 (needs vLLM)
```

## Migration notes

Phiên bản trước có hai layer riêng: `gear/spo/` (SPO) + `gear/gear_src/` (GEAR ext). Refactor này gộp tất cả vào `treetune/` ở top level:

| Cũ | Mới |
|----|-----|
| `gear/spo/src/treetune/` | `gear/treetune/` |
| `gear/spo/src/guidance/` | `gear/guidance/` |
| `gear/gear_src/gear_ext/core/` | `gear/treetune/gear/` |
| `gear/gear_src/gear_ext/episode_generators/gear_episode_generator.py` | `gear/treetune/episode_generators/gear_episode_generator.py` |
| `gear/gear_src/gear_ext/inference_strategies/gear_inference_strategy.py` | `gear/treetune/inference_strategies/gear_inference_strategy.py` |
| `gear/spo/configs/*` + `gear/configs/*` | `gear/configs/*` (merged) |
| `gear/spo/scripts/*` + `gear/scripts/*` | `gear/scripts/*` (merged) |
| `gear_main.py` shim | xoá — dùng `python -m treetune.main` |
| `import gear_ext.X` | `import treetune.gear.X` |
| `setup.py` (gear_ext) | `setup.py` (treetune, single package) |

DAPO không có trong scope — dùng GRPO (`scripts/train_grpo_MATH.sh`) thay thế.

## License

Code base treetune kế thừa MIT License của SPO. Xem `docs/legacy/LICENSE_SPO`.

## Tham khảo

- [SPO paper](https://github.com/AIFrameResearch/SPO) — Segment Policy Optimization gốc
- [GRPO paper](https://arxiv.org/abs/2402.03300) — DeepSeek-Math
- [VinePPO paper](https://arxiv.org/abs/2410.01679) — McGill
- [DPO paper](https://arxiv.org/abs/2305.18290) — Stanford
- `PLAN.md` — đặc tả chi tiết thuật toán GEAR

### Benchmark được đánh giá trong lúc training

Chuẩn bị các dataset eval local:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning
python scripts/download_eval_datasets.py
```

Các cấu hình training trên MATH chạy thêm năm benchmark ở mỗi lần evaluation:

- `aime24_test`: `math-ai/aime24` (30 bài).
- `aime25_test`: `math-ai/aime25` (30 bài).
- `amc23_test`: `math-ai/amc23` (40 bài AMC 2023).
- `olympiadbench_test`: file
  `OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet` của
  `Hothan/OlympiadBench` (674 bài).
- `collegeMath_test`: `data/collegeMath` (2,818 bài).

Downloader pin revision Hugging Face và lưu từng nguồn ở dạng `DatasetDict` dưới
`data/`. Thư mục cũ `data/olympiadbench` có 675 bài, gồm một bài không còn trong
bản Hugging Face hiện tại và 15 đề còn placeholder lỗi; benchmark dùng bản chuẩn
ở `data/olympiadbench_hf`.

Smoke test format prompt và generation bằng model nhỏ:

```bash
python scripts/smoke_eval_datasets.py --local-files-only
```

Tần suất evaluation vẫn được điều khiển bởi `GEAR_EVAL_EVERY_N_ITERATIONS`.
