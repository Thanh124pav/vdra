# Migrate treetune (DeepSpeed) → verl framework

> Kế hoạch refactor/migration của dự án. File `PLAN.md` ở gốc repo là **đặc tả thuật
> toán GEAR** (công thức, Lemma), KHÔNG phải kế hoạch migration — đừng nhầm hai file.
> Bản gốc do Claude Code tạo ở `~/.claude/plans/`; file này là bản commit vào repo.

## Progress log (living)

**Decisions chốt khi thực thi:**
- verl pinned to **v0.6.0** (bản mới nhất có CẢ synchronous SPMD `vLLMRollout.generate_sequences`
  cho tree branching kiểu TreePO VÀ registry `register_policy_loss`/`register_adv_est`). v0.3.1–0.4.x
  thiếu registry; v0.7+ bỏ SPMD (PR #4411). Old checkout giữ ở `verl_0.8.0.dev_backup/` (reversible).
- Dedicated env **`verl060`** (conda, python 3.10, transformers==4.48.3 vì ≥4.49 bỏ
  `AutoModelForVision2Seq`; torch 2.6 cpu; tensordict 0.10; ray/hydra/omegaconf). Env `deeplearning`
  (transformers 5.x) chỉ dùng cho treetune gốc + parity đối chiếu.

**Done + verified (CPU/no GPU):**
- ✅ **Env `verl060` CPU-test** - `setup_env.sh` reaches `ENV_SETUP_DONE`; dependency set is scoped to recipe import/unit tests to avoid the `datasets/httpx` resolver path.
- ✅ **Step 0** - vendored `verl/recipe/gear_tree/gear_core/` (gear + tree_update_modes + grading + logging_utils). Existing vendor parity remains the baseline.
- ✅ **Step 5** - `policy_loss.py` registers `treetune_ppo`, preserves treetune PPO math, and now returns the verl v0.6.0 policy-loss contract `(loss, clipfrac, kl, clipfrac_lower)`.
- ✅ **Step 1 (CPU glue)** - `reward.py` registers `gear_math`, uses the vendored grader, and writes scalar reward to the last valid response token.
- ✅ **Step 4 (CPU glue)** - `tree_advantage.py` ports tree edge extraction and broadcasts edge scalars into `DataProto.batch["advantages"]`, `returns`, `token_level_rewards`, and optional `old_log_probs`. Now also carries `query_token_ids`.
- ✅ **Step 2 core (CPU-tested)** - `tree_rollout.py` `build_tree` mirrors treetune `_construct_tree` byte-for-byte (segment@M, finish_reason leaf/expand, reward=mean(children)/std). Golden-numerics test compares against an inline transcription of `_construct_tree` (`test_tree_rollout.py`). `reward_function.py` ports `MATHRewardFunction.__call__` (extraction + unfinished/multi-#### penalties + grade, both MATH/minerva modes). `tree_data.py` assembles edges -> verl `DataProto` (prompt left-pad / response right-pad). `vllm_rollout_tree.py:build_edge_batch` unit-tested with a fake vLLM engine.
- ✅ **Step 3 core (CPU-tested)** - `gear_gate.py` `GearGate`: `simple` predict-k perplexity **prune** ported byte-exact (`k=ceil(exp(-sum_logprobs/num_tokens))`, clamp `[n_min, default_bf]`, root/near-leaf skip). Sibling-local **share** + TV-variance budget paths wired to `gear_core` behind an injected vLLM logprob scorer (GPU); safe no-op without it.
- ✅ **Pipeline integration (CPU)** - `test_pipeline_integration.py`: mock engine -> build_tree -> edges -> DataProto -> `treetune_ppo` loss, verifying shapes + per-token advantage placement. 22 passed / 1 skipped on `verl060`.

**Config/algorithm changes + run scripts + logging (this round):**
- ✅ **Config defaults** — `root_allocation=true`, `n_tv_estimates=null`→auto `branch_factor**2` (`GearGate.n_tv_estimates_for`), KL coef 0 everywhere (`algorithm.kl_ctrl.kl_coef`, `actor.kl_loss_coef`, `use_kl_in_reward=false`), `budget_lambda=0`.
- ✅ **Allocation formula change (intentional, per user)** — `budget_allocation.py` now weights `sqrt(sigma^2 - lambda)` (was `sqrt(sigma^4 - lambda)`), default `lambda=0`. Documented as a deliberate deviation from treetune; `test_logging_and_alloc.py` locks the new behaviour.
- ✅ **11 algorithms, run scripts** — `scripts/train_{grpo,rloo,vineppo,spo_chain,spo_tree,treerl,treepo,gear_spo_chain,gear_spo_tree,gear_treerl,gear_treepo}.sh` + `_common.sh`. GRPO/RLOO route through verl-native `main_flat.py` (exact built-in estimators + `treetune_ppo` loss + `gear_math`); the rest through the tree recipe. All `bash -n` clean.
- ✅ **Logging (treetune parity)** — `tree_logging.py` `TreeDemoLogger` reuses vendored `logging_helpers`/`tree_policy_logging` → `gear_demos/demos.jsonl` + `demos.md` (per-tree stats, per-depth counts, SHARE/PRUNE/budget demo rows) + `full_trees/tree_N.json` (one complete example tree, rate-limited) + console stats. Trainer writes `training_timing.jsonl` (per-step generation/update/wall). **Verified on GPU**: real run produced 2 demo records + 2 full-tree dumps + per-tree stats.

**GPU-validated (real model, HF generation, no Ray/FSDP):**
- ✅ **Real-model smoke** — `scratchpad/smoke_gear_tree_hf.py`: SmolLM2-135M on GPU → native tree rollout (real per-token logprobs) → MATH grading → SPO/GEAR advantages → `edges_to_dataproto` → `treetune_ppo` loss. Loss numerics verified (`approx_kl == 0.5·Δ²`). vLLM path NOT used: this env's vllm 0.22 has a broken cu13 build (engine-core subprocess crash) — the `vLLMTreeRollout` binding is code-correct but unrunnable here; HF `model.generate` drives the engine-agnostic `segment_fn` instead.
- ✅ **GEAR-VinePPO** — `vineppo_advantage.py` ports `_compute_mc_value` + `_compute_step_advantages` byte-faithful; `build_edge_batch(vineppo_K>0)` swaps internal-node values for K-rollout MC estimates. CPU-tested.
- ✅ **GEAR LP scorer** — `engine_scorer.py` `EngineLPScorer` mirrors `LPScorer.score_one` against the offline vLLM engine (`prompt_logprobs`), wired into `GearGate.scorer` when `gear.enable_share`. CPU-tested with a fake engine.

**Code-complete, needs GPU + Ray to validate E2E:**
- ⏳ **Step 2/3 worker** - `gear_tree_worker.py` `GearTreeActorRolloutWorker` adds dispatched `build_trees` (mirrors `generate_sequences` mode-switch), re-classes the SPMD rollout to `vLLMTreeRollout`.
- ⏳ **Step 6** - `gear_ray_trainer.py` `RayGearTreeTrainer(RayPPOTrainer)` overrides `fit()`: prompts -> `build_trees` (tree rollout + precomputed advantages) -> `compute_log_prob` (old logp) -> `update_actor` (`treetune_ppo`). Bypasses verl `compute_advantage`.
- ⏳ **Step 7** - `main_gear_tree.py` (TaskRunner + `RayGearTreeTrainer`), `config/gear_tree_trainer.yaml` (Hydra compose verified), `run_gear_tree.sh` (9 variants via CLI overrides).

**Next:** Full Ray+FSDP+vLLM E2E training run (needs a working vLLM install — the
bundled `verl060` is CPU-only and `deeplearning`'s vllm 0.22 has a broken cu13
build; a clean verl-0.6.0 + vllm env on a bigger GPU is required). Golden-numerics
parity vs treetune on a fixed problem/seed (Verification #2, needs treetune runnable).
Step 8: remove treetune/DeepSpeed once parity is signed off.

## Context

Codebase hiện tại (`treetune/`) là một framework RL cho reasoning LLM chạy trên **DeepSpeed**,
với đóng góp nghiên cứu chính là **GEAR** (Information-Gated Policy Optimization: online
segment-level pruning/sharing cho SPO-tree). Mục tiêu: **port sang verl** và cuối cùng **thay thế**
layer treetune/DeepSpeed, giữ **nguyên vẹn 100% logic thuật toán** (yêu cầu bắt buộc của người dùng).

Quyết định đã chốt với người dùng:
- **Phạm vi**: port họ tree + GEAR: `SPO-chain`, `SPO-tree`, `TreeRL`, `TreePO`, `GEAR-SPO-chain`,
  `GEAR-SPO-tree`, `GEAR-VinePPO`, `GEAR-TreeRL`, `GEAR-TreePO`. (PPO/GRPO/RLOO/DPO/RestEM **không**
  trong scope — GRPO/RLOO/PPO đã có sẵn built-in trong verl.)
- **Tree generation**: **viết lại native trên verl rollout** (không tái dùng inference code cũ),
  học theo repo tham chiếu **TreePO** (`recipe/treepo/vllm_rollout_tree.py`, `DataSampleTree`).
- **Deliverable**: dần **thay thế** treetune bằng verl.

### Bản đồ kiến trúc (đã khảo sát)

**treetune loop** (`treetune/runtime/policy_iteration_runtime.py:257`):
`for iteration: episodes = episode_generator.generate(); trainer.step(episodes)`.
Toàn bộ tính mới nằm ở **episode generation** (dựng cây SPO/GEAR + prune/share + tính advantage);
DeepSpeed PPO trainer chỉ **tiêu thụ per-token advantages đã tính sẵn**.

**Điểm mấu chốt giúp port khả thi**: `Episode` dataclass
(`treetune/episode_generators/base_episode_generator.py:21`) mang
`advantages` (per-response-token, đã tính sẵn), `values`, `actor_shifted_log_probs`, `scores`.
Trong verl, đây tương ứng với việc ghi thẳng vào `DataProto.batch["advantages"]` / `["old_log_probs"]`.

**Các module thuật toán thuần Python, KHÔNG phụ thuộc DeepSpeed → vendor nguyên xi (byte-identical)**:
- `treetune/gear/*` (budget_allocation, tv_distance, tv_estimators, thresholds, log_prob_matrix,
  segment_index, local_value_share, online_budget, pruning_controller, triggers, lp_scorer, vllm_scorer)
- `treetune/episode_generators/tree_update_modes.py` (`compute_tree_update_values` — advantage core)
- `treetune/tasks/math_grader.py`, `math_grader_minerva.py` (grading thuần)

**verl target** (pinned **v0.6.0** tại `verl/`): loop `RayPPOTrainer.fit()`
(`verl/verl/trainer/ppo/ray_trainer.py`):
generate → reward → old_log_prob → ref_log_prob → values → `compute_advantage` → critic → actor.
Seam mở rộng: `@register_adv_est`, `@register_policy_loss`, custom reward manager,
`RayPPOTrainer` subclass + `TaskRunner`, `DataProto` (`non_tensor_batch["uid"]` = group id).

---

## ⚠️ Quyết định verl version cho tree rollout — ĐÃ RESOLVED (v0.6.0)

Embedded verl ban đầu là 0.8.0.dev, đã **bỏ SPMD rollout (PR #4411)** — generation đi qua async vLLM
server. **TreePO** dựng cây bằng cách **extend SPMD rollout đồng bộ** (`vllm_rollout_spmd.py`). Sinh
segment-by-segment + branch cần điều khiển engine vLLM đồng bộ — async abstraction giấu mất khả năng này.

**Đã chốt: pin verl `v0.6.0`** — bản mới nhất còn `vllm_rollout_spmd.py` (`class vLLMRollout(BaseRollout)`,
synchronous `generate_sequences`) VÀ đã có registry `register_policy_loss`/`register_adv_est`, đường dẫn
`verl.workers.config.ActorConfig` tương thích. Vừa giữ được cơ chế dựng cây kiểu TreePO, vừa dùng được
extension point sạch. (v0.3.1–0.4.x còn SPMD nhưng chưa có registry → sẽ phải monkeypatch xấu hơn.)

---

## Kế hoạch triển khai

Tạo recipe mới `verl/recipe/gear_tree/` (không đụng core verl trừ 1 registry import). Cấu trúc phân
tầng đúng theo cách treetune factorize: **(rollout cây) × (GEAR gate tùy chọn) × (tree_update_mode advantage)**.

### Bước 0 — Chuẩn bị verl + vendor module thuần  ✅ DONE
- Pin verl v0.6.0 vào `verl/` (backup bản cũ ở `verl_0.8.0.dev_backup/`).
- Vendor **nguyên văn** vào `verl/recipe/gear_tree/gear_core/`:
  `treetune/gear/*.py`, `tree_update_modes.py`, `tasks/math_grader*.py`. Chỉ sửa `import` path,
  **không đổi công thức**. `tests/test_vendor_parity.py` đối chiếu bản gốc ↔ vendored.

### Bước 1 — Reward manager (grading)
- `verl/recipe/gear_tree/reward.py`: subclass `AbstractRewardManager`
  (`verl/verl/workers/reward_manager/abstract.py`) hoặc `compute_score` fn, gọi
  `math_grader.grade_answer` / `eval_math` đã vendor. Ghi scalar reward ở token response cuối
  (mẫu `naive.py`). Đăng ký `@register("gear_math")`.
- Lưu ý: trong tree rollout, **leaf reward được chấm ngay trong lúc dựng cây** (để tính mean-reward
  subtree → segment advantage). Reward manager verl ở đây chủ yếu là passthrough/validation vì
  advantage đã tính trong generation.

### Bước 2 — Native tree rollout (theo TreePO)
- `verl/recipe/gear_tree/vllm_rollout_tree.py`: dựng cây segment native, học `DataSampleTree` +
  segment stepping của TreePO. Mỗi node = segment: sinh tối đa `M` tokens (default 100), `finish_reason
  == "length"` ⇒ expandable, ngược lại ⇒ leaf → chấm reward. Chain = branch factor 1; tree = branch
  factor B theo depth (`branch_factor_<shape>`, ví dụ 666/4444). Bảo toàn: reward node =
  `mean(child_rewards)`, `reward_std = std(...)` (mẫu `hybrid_inference_strategy.py:443`).
- Extend vLLM SPMD rollout worker (mẫu TreePO extend `vllm_rollout_spmd.py`). Cần vLLM trả
  **per-token logprobs** cho mỗi segment (SamplingParams `logprobs`) để phục vụ GEAR.
- Output: cấu trúc cây per-prompt (node: `text/full_text/depth/reward/sum_logprobs/num_tokens/children`).

### Bước 3 — GEAR gate (overlay tùy chọn, dùng module đã vendor)
Bật khi thuật toán có tiền tố `GEAR-`. Móc vào vòng expand từng depth của Bước 2, **giữ nguyên logic**:
- **Predict-k / prune redundant siblings**: `tv_estimators.estimate_k_for_parent` (union-find gom
  prefix TV < `epsilon`), trả `predicted_k` + candidates.
- **Budget allocation theo variance**: `budget_allocation.allocate_branch_factors` với trọng số
  `sqrt(sigma_i^4 - budget_lambda)`, floor `n_min`; reserve-pool recycle (`online_budget.py`,
  `SharedReservePool`/`RootQueueManager`) — chính là cơ chế đạt `O((ρW)^D)`.
- **Sibling-local value share**: `segment_index.SegmentBST.find_nearest` (key = AvgLP_K) +
  `local_value_share`; threshold `τ = η + sqrt(log(2/α)/(2K))`, `η` từ Lemma 2.4 (`thresholds.py`).
- **Logprob scoring** cho answer-set Y: `vllm_scorer.VLLMLogprobClient` (`/completions`,
  `echo=True, logprobs=1, max_tokens=0`) + `lp_scorer.LPScorer`. Trong verl, trỏ tới vLLM engine
  của rollout worker (cùng endpoint) — cần đảm bảo phơi được completions+echo (điểm cần verify sớm).
- `skip_near_leaf_expand`, `root_allocation` giữ đúng semantics `gear_defaults.libsonnet`.

### Bước 4 — Tính advantage (tree_update_mode) + nạp vào DataProto
- `verl/recipe/gear_tree/tree_advantage.py`: từ cây → edges (mẫu `tree_episode_generator.py:113` và
  GEAR override `gear_episode_generator.py:116`), gọi `compute_tree_update_values` đã vendor:
  - `spo` → advantage = `child_reward - parent_reward` (SPO/GEAR-SPO)
  - `treepo_original` → `(1-gw)·local + gw·global` (TreePO/GEAR-TreePO)
  - `treerl_original` → value = `imm + γ·child_reward`, adv = `value - parent_reward` (TreeRL/GEAR-TreeRL)
  - **VinePPO** (GEAR-VinePPO): MC-value TD advantage — port `_compute_mc_value` +
    `_compute_step_advantages` (`vineppo_episode_generator.py:377`) thành 1 nhánh trong rollout.
  - GEAR share/prune edge: reward inherit target; pruned ⇒ `advantage=0` (nếu emit), else drop
    (`zero_advantage_when_pruned`).
- **Broadcast scalar → per-token** (mẫu `hybrid_episode_generator.py:632`):
  `advantages[i] = edge.advantage` cho mọi response token. Ghi thẳng vào
  `batch.batch["advantages"]` (+ `values` nếu dùng). Bỏ qua verl `compute_advantage` mặc định.
  Group id (`uid`) gán per-prompt để log/diagnostics.

### Bước 5 — Custom PPO policy loss (giữ đúng numerics)  ✅ DONE
- `@register_policy_loss("treetune_ppo")` (`verl/recipe/gear_tree/policy_loss.py`), **sao chép chính xác**
  từ `treetune/trainers/ppo_trainer.py:1070`:
  - `use_prob_mask`: loại token có `exp(old_logp) ≥ 0.9` khỏi loss.
  - `log_ratio = (logp - old_logp)·mask`; **clamp ±10** rồi `exp` (verl mặc định clamp ±20).
  - `pg = max(-A·ratio, -A·clamp(ratio,1±cliprange))`, `masked_mean`. **ratio_threshold skip**.
  - logits `/= temperature` ở mọi forward (config actor). Whiten/grayen advantages làm ở Bước 4.
- **KL**: KL-in-reward (`apply_kl_penalty`) / KL-in-loss (`use_kl_loss`) — mặc định tree config tắt cả hai.
- **Critic** (optional): value loss hệ số cứng `0.5`, return = discounted MC return.

### Bước 6 — Trainer subclass + entry point
- `verl/recipe/gear_tree/gear_ray_trainer.py`: `RayGearTreeTrainer(RayPPOTrainer)` override phần
  generate+advantage của `fit()`: thay `generate_sequences` bằng tree rollout (Bước 2) + GEAR (Bước 3)
  + tree advantage (Bước 4) → nạp `advantages`/`old_log_probs` vào `DataProto`, rồi gọi actor/critic
  update chuẩn của verl (dùng `treetune_ppo` loss).
- `verl/recipe/gear_tree/main_gear_tree.py`: `TaskRunner` + `run_ppo(config, task_runner_class=...)`
  (mẫu `main_ppo.py` và pattern recipe DAPO). Import module đăng ký adv/loss/reward trước khi train.

### Bước 7 — Config (Jsonnet → Hydra YAML) + scripts
- `verl/recipe/gear_tree/config/`: `gear_tree_trainer.yaml` compose từ verl base + block `algorithm`,
  `actor.policy_loss.loss_mode=treetune_ppo`, `rollout.n`/tree shape, và block `gear` map từ
  `configs/gear_defaults.libsonnet` (epsilon, budget_lambda, n_min, n_tv_estimates, k_algorithm,
  skip_near_leaf_expand, root_allocation, tree_update_mode, ...).
- Mỗi biến thể 1 overlay YAML (9 thuật toán). Scripts `run_<algo>_<dataset>.sh`.

### Bước 8 — Gỡ layer treetune/DeepSpeed (giai đoạn cuối)
Sau khi verl path chạy + đối chiếu numerics đạt: gỡ dần `treetune/trainers/deepspeed_*`,
`ppo_trainer.py`, runtime DeepSpeed, `configs/deepspeed`, `scripts/train_*.sh` cũ, và
`verl_0.8.0.dev_backup/`. Giữ lại (đã vendor vào recipe) phần math + grading. README cập nhật.

---

## Files chính sẽ tạo/sửa

| File | Vai trò | Trạng thái |
|------|---------|------------|
| `verl/recipe/gear_tree/gear_core/*` | vendor gear math + tree_update_modes + grading | ✅ done |
| `verl/recipe/gear_tree/policy_loss.py` | `@register_policy_loss("treetune_ppo")` | ✅ done |
| `verl/recipe/gear_tree/vllm_rollout_tree.py` | native tree/segment rollout (theo TreePO) | ⏳ |
| `verl/recipe/gear_tree/gear_gate.py` | GEAR online prune/share + budget allocation | ⏳ |
| `verl/recipe/gear_tree/tree_advantage.py` | edges -> advantage -> per-token -> DataProto | ✅ CPU glue done |
| `verl/recipe/gear_tree/reward.py` | reward manager calls vendored grading | ✅ CPU glue done |
| `verl/recipe/gear_tree/gear_ray_trainer.py` | `RayGearTreeTrainer(RayPPOTrainer)` | ⏳ |
| `verl/recipe/gear_tree/main_gear_tree.py` | entry point + TaskRunner | ⏳ |
| `verl/recipe/gear_tree/config/*.yaml` | Hydra config cho 9 biến thể | ⏳ |
| `verl/recipe/gear_tree/run_*.sh` | training scripts | ⏳ |

## Nguyên tắc bảo toàn thuật toán (bắt buộc)
1. Mọi công thức (advantage, TV, threshold, budget, PPO loss, KL) **copy nguyên** từ treetune, chỉ
   đổi glue/IO. Ưu tiên **vendor & import**, hạn chế viết tay lại math.
2. Chỗ **bắt buộc viết lại** = tree rollout orchestration (vì đổi engine gen). Đây là điểm rủi ro
   lệch logic cao nhất → phải verify từng bước.

## Verification (end-to-end)

1. **Unit đối chiếu math**: port toàn bộ `tests/test_*` liên quan gear (test_budget_allocation,
   test_tv_distance, test_thresholds, test_tree_update_modes, test_local_value_share,
   test_log_prob_matrix, test_segment_index, test_online_gear, test_gear_algorithm_variants) chạy
   trên module đã vendor — **phải pass y hệt**. (đã có `test_vendor_parity.py` + `test_policy_loss_parity.py`)
2. **Golden numerics rollout→advantage**: với 1 problem + seed cố định, so **tree structure + segment
   rewards + per-token advantages** giữa treetune và verl recipe. Khớp tới sai số float. Gate quan
   trọng nhất cho quyết định "viết lại native".
3. **PPO loss parity**: cùng batch → so `pg_loss`/`clip_frac`/`approx_kl` giữa `treetune_ppo` và
   `_compute_actor_loss` gốc. ✅ done.
4. **Smoke E2E**: `run_gear_tree_MATH.sh` với model nhỏ (SmolLM2-135M), depth 2, 2 iterations,
   xác nhận chạy full loop generate→reward→adv→update→save trên verl, log GEAR demos.
5. **Prune/share rate + reward variance** (Exp 2 của PLAN.md) khớp xu hướng treetune.
