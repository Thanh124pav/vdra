# Tree Segmentation Và GEAR Allocation

Tài liệu này mô tả cách repo tách phần xây cây khỏi phần GEAR prune/allocate. Điểm chính: contribution prune và allocate không phụ thuộc vào cách chia segment của SPO. Cùng một controller có thể chạy trên cây được tạo bởi `spo_step`, `treepo_fixed_step`, `treerl_entropy`, hoặc một builder khác miễn là node có `text`, `full_text`, `children` và metadata depth/segment tương đương.

## Khái Niệm

- `segmentation` / `builder`: cách tạo frontier node. Ví dụ `spo_step` chia theo bước reasoning kiểu SPO, `treepo_fixed_step` chia theo chunk cố định, `treerl_entropy` ưu tiên vùng entropy cao.
- `k_algorithm`: thuật toán ước lượng số nhánh cần giữ tại một node. Mặc định hiện tại là `hierarchical`.
- `pruning`: quyết định duplicate/prune/keep dựa trên probability evidence, TV distance, value gap và variance bound.
- `allocation`: phân phối lại budget nhánh giữa các node/frontier sau khi đã có evidence.

## Backward Compatibility

Config GEAR cũ vẫn map về:

```text
segmentation=spo_step
allocation=gear
k_algorithm=hierarchical
```

`scripts/train_gear_tree_MATH.sh` tiếp tục dùng pipeline training hiện tại. Các knob mới chỉ thêm logging và manifest, không đổi default pruning/allocation runtime.

## Cấu Hình Cây

- `TREE` hoặc `GEAR_TREE`: shape cây, ví dụ `666` nghĩa là 3 tầng, mỗi node mặc định sinh 6 children.
- `TREE_M` hoặc `GEAR_TREE_M`: token budget cho tree rollout.
- `branch_factors`: bản mở rộng từ shape, ví dụ `666` thành `{0: 6, 1: 6, 2: 6}`.
- `GEAR_FULL_TREE_DEMO_EVERY_N_TREES`: log thêm full tree theo chu kỳ.
- `GEAR_FULL_TREE_DEMO_MAX_TREES`: số full tree đầu tiên luôn log.

Khi training, terminal in manifest dạng:

```text
[tree-policy] mode=training training=true
[tree-policy] algorithm=gear_spo
[tree-policy] segmentation=spo_step
[tree-policy] allocation=budget_allocation
[tree-policy] pruning=true
[tree-policy] tree_shape=666 tree_m=600 depth=3 branch_factors={0:6,1:6,2:6}
[tree-policy] k_algorithm=hierarchical residual_budget=true root_allocation=true
```

Artifact full-tree nằm trong `<exp>/gear_demos/`:

- `run_manifest.json`
- `full_trees.jsonl`
- `full_trees.md`

Full-tree logging có rate limit, nhưng khi đã log thì giữ nguyên `text`, `full_text`, `children`, segmentation metadata và evidence, không truncate.


## Biến Thể GEAR Cho Baseline Khác

Repo có thêm hai entrypoint training:

```bash
# GEAR trên SPO-chain, mặc định TREE=6.
bash scripts/train_gear_spo_chain_MATH.sh

# GEAR trên VinePPO, mặc định TREE=6 và hiện chỉ hỗ trợ TREE một chữ số.
bash scripts/train_gear_vineppo_MATH.sh
```

`GEAR-SPO-chain` dùng `gear_episode_generator` và shared `gear` inference strategy. `GEAR-VinePPO` dùng `gear_vineppo_episode_generator`: phần trajectory/advantage vẫn là VinePPO, còn rollout tree do GEAR controller xây để có pruning/allocation evidence và full-tree artifact.

Theo default, tất cả biến thể GEAR (`gear_tree`, `gear_spo_chain`, `gear_vineppo`) đều dùng `gear_k_algorithm=hierarchical`, `gear_allocation_mode=budget_allocation`, `gear_use_residual_budget=true` và `gear_root_allocation=true`. Vì vậy mọi biến thể đều có bước dự đoán `k`, pruning khi `k < n`, và allocation khi `k >= n`; `prune_only` chỉ là ablation opt-in.

## Prunning Analysis Không Training

Folder `prunning_analysis/` chỉ dùng để chứng minh hiệu quả prune. Script không gọi trainer, không gọi `run_iteration_loop`, không gọi DeepSpeed, không ghi checkpoint, không update policy weights.

Chạy replay không cần vLLM:

```bash
python prunning_analysis/run_prunning_analysis.py \
  --backend replay \
  --input-tree path/to/full_trees.jsonl \
  --builder spo_step
```

Chạy local Transformers:

```bash
python prunning_analysis/run_prunning_analysis.py \
  --backend transformers \
  --builder treepo_fixed_step \
  --model sshleifer/tiny-gpt2
```

Chạy live vLLM:

```bash
export APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1
python prunning_analysis/run_prunning_analysis.py \
  --backend vllm \
  --builder treerl_entropy \
  --model /path/to/model
```

## Cách Đọc Evidence

- `p_x`, `p_y`: phân phối probability/logprob evidence của hai child cần so sánh.
- `|v_x-v_y|` (`value_gap` trong JSON): độ chênh value/reward giữa hai child nếu có.
- `tv`: total variation distance giữa `p_x` và `p_y`; TV nhỏ thường là tín hiệu duplicate.
- `value_upper_bound`: bound suy ra từ TV theo threshold config, dùng để so với `|v_x-v_y|`.
- `reward_variance_sigma2`: variance estimate từ các pair TV tại node.
- `sigma4`: bình phương của variance estimate, tiện cho lập luận bound/ổn định.
- `duplicate`, `prune_candidate`, `keep`: quyết định cuối cùng của controller cho pair đó.

Nếu replay tree thiếu `prob_matrix` hoặc `pair_tvs`, script ghi rõ `unavailable_fields` và không fake số.

## Artifact Để Viết Paper

Mỗi run ghi vào `prunning_analysis/outputs/<run>/`:

- `run_manifest.json`: backend, builder, tree shape, k algorithm, training=false.
- `prunning_trace.jsonl`: log từng node/pair với đầy đủ p, TV, upper bound, variance và quyết định prune.
- `prunning_summary.json`: tỷ lệ prune candidate, duplicate, TV mean/max, variance mean/max.
- `full_tree_before.json`: cây gốc trước k_algorithm.
- `full_tree_after_k_algorithm.json`: cây sau khi annotate quyết định k/prune.
- `report.md`: bản đọc nhanh cho paper/debug.

Để chứng minh pruning hiệu quả, dùng `prunning_trace.jsonl` cho bảng định lượng và `full_tree_before.json` / `full_tree_after_k_algorithm.json` để đối chiếu text đầy đủ giữa các thuật toán builder.

## Update objective options

Mặc định training vẫn dùng `tree_update_mode: 'spo'`, tức advantage của một
cạnh cây là tín hiệu local hiện có: `reward(child) - reward(parent)`. Điều này
giữ backward compatibility cho các script cũ.

Để chạy ablation theo phong cách TreePO/TreeRL và so sánh trong cùng stack,
có thể đổi objective mà không đổi builder/segmentation. Các mode này là
style/parity ablation, không phải claim tái hiện chính thức y hệt paper gốc:

- `spo`: local segment advantage hiện tại.
- `treepo_original`: kết hợp local segment advantage với global advantage từ
  root tới node hiện tại. Trọng số global được điều khiển bởi
  `treepo_global_weight` (default `0.5`).
- `treerl_original`: dùng dense process/TD-style target trên từng cạnh cây với
  `treerl_gamma` (default `0.9`).

Ví dụ chạy GEAR-tree với objective TreePO-style:

```bash
GEAR_TREE_UPDATE_MODE=treepo_original \
GEAR_TREEPO_GLOBAL_WEIGHT=0.5 \
bash scripts/train_gear_tree_MATH.sh
```

Ví dụ chạy objective TreeRL-style:

```bash
GEAR_TREE_UPDATE_MODE=treerl_original \
GEAR_TREERL_GAMMA=0.9 \
bash scripts/train_gear_tree_MATH.sh
```

Các ablation config tương ứng nằm ở:

- `configs/ablations/abl_treepo_original_update.jsonnet`
- `configs/ablations/abl_treerl_original_update.jsonnet`

Các mode này chỉ thay cách tạo `advantages`/`value` cho PPO trainer. Builder
cây, segmentation, pruning và allocation vẫn là các option độc lập.
