# Prunning Analysis

`prunning_analysis/` là pipeline analysis-only để chứng minh pruning thực sự hiệu quả. Pipeline này không training, không gọi trainer, không gọi `run_iteration_loop`, không gọi DeepSpeed, không ghi checkpoint và không update policy weights.

## Backend

- `--backend replay`: chạy local/offline. Đọc cây đã lưu từ JSON/JSONL, hoặc dùng `--synthetic-replay` để tạo cây deterministic có probability evidence cho smoke test.
- `--backend transformers`: chạy local không cần vLLM. Dùng HuggingFace Transformers từ `--model` để score probability. Model phải có sẵn trong cache hoặc môi trường phải tải được model.
- `--backend vllm`: chạy live với vLLM endpoint. Dùng `--api-base` hoặc `APP_OPENAI_VLLM_API_BASE` trỏ tới OpenAI-compatible `/completions` endpoint.

## Builder / Segmentation

Script hỗ trợ các builder sau:

- `spo_step`: chia theo step reasoning kiểu SPO.
- `treepo_fixed_step`: chia theo fixed-size step/chunk, phù hợp so sánh với TreePO-style rollout.
- `treerl_entropy`: mô phỏng segmentation theo vùng entropy cao, phù hợp so sánh với TreeRL-style expansion.

GEAR controller trong analysis chỉ cần frontier/tree node có `text`, `full_text`, `children`, depth/segment metadata và probability evidence nếu chạy replay. Nó không phụ thuộc field riêng của SPO.

## Ví Dụ Chạy

```bash
# Smoke run local, không vLLM, không training.
bash scripts/run_prunning_analysis.sh --synthetic-replay

# Replay artifact full-tree đã có.
bash scripts/run_prunning_analysis.sh \
  --backend replay \
  --input-tree experiments/run/gear_demos/full_trees.jsonl

# Local Transformers scorer.
BACKEND=transformers MODEL=sshleifer/tiny-gpt2 \
bash scripts/run_prunning_analysis.sh

# vLLM scorer.
BACKEND=vllm MODEL=Qwen/Qwen3-0.6B \
APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1 \
bash scripts/run_prunning_analysis.sh
```

Run banner luôn in rõ:

```text
[prunning-analysis] mode=analysis_only training=false
[prunning-analysis] backend=replay
[prunning-analysis] builder=spo_step
[prunning-analysis] k_algorithm=hierarchical
[prunning-analysis] allocation=false pruning=true
[prunning-analysis] tree_shape=666 tree_m=600
```

## Artifact

Mỗi run ghi vào `prunning_analysis/outputs/<run>/`:

- `run_manifest.json`: backend, builder, tree shape, token budget, k algorithm, allocation=false, pruning=true, training=false.
- `prunning_trace.jsonl`: một record cho mỗi node/pair decision.
- `prunning_summary.json`: thống kê aggregate về prune candidate, duplicate, TV và variance.
- `full_tree_before.json`: cây đầy đủ trước khi annotate k_algorithm.
- `full_tree_after_k_algorithm.json`: cây sau khi annotate `gear_predicted_k`, duplicate/prune counts.
- `report.md`: bản đọc nhanh gồm manifest, summary và trace preview.

Artifact giữ nguyên `text`, `full_text`, `children`, segmentation metadata và probability/TV/variance evidence. Analysis có thể limit số node/pair được inspect, nhưng đã ghi tree thì không truncate text.

## Cách Đọc Trace

- `p_x`, `p_y`: phân phối xác suất của hai child trên cùng support.
- `|v_x-v_y|` / `value_gap`: chênh lệch value hoặc reward giữa hai child nếu replay tree có field này.
- `tv`: total variation distance giữa `p_x` và `p_y`.
- `value_upper_bound`: upper bound suy ra từ TV theo threshold config.
- `reward_variance_sigma2`: variance estimate tại node từ các pair TV.
- `sigma4`: bình phương của variance estimate.
- `duplicate`: pair có TV thấp hơn duplicate threshold.
- `prune_candidate`: pair đủ điều kiện là ứng viên prune.
- `keep`: quyết định giữ pair/nhánh trong diagnostic.

Nếu replay tree thiếu `prob_matrix` hoặc `pair_tvs`, trace sẽ ghi `unavailable_fields` và không fake số.
