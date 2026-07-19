#!/usr/bin/env bash
# PLAN.md P0.8 — one pre-GPU gate command.
#
# Runs every CPU-verifiable invariant that must hold before launching a GPU
# smoke run and prints exactly one line on success:
#
#     PRE_GPU_CHECK=PASS
#
# Any failure returns a non-zero exit code and prints nothing after that
# point.

set -euo pipefail

# Resolve the repo root regardless of caller cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

log() {
    printf "[pre_gpu_check] %s\n" "$*" >&2
}

fail() {
    printf "[pre_gpu_check] FAIL: %s\n" "$*" >&2
    exit 1
}

log "1/12 compileall vdra_core and gear_tree"
python -m compileall -q vdra_core verl/recipe/gear_tree \
    || fail "compileall failed"

log "2/12 targeted CPU tests (root)"
python -m pytest tests/ -q \
    --ignore=tests/test_math_competition_evaluation.py \
    --ignore=tests/test_vllm_launch_scripts.py \
    --ignore=tests/test_vllm_scorer.py \
    --ignore=tests/test_vllm_server.py \
    --ignore=tests/test_gear_algorithm_variants.py \
    --ignore=tests/test_gear_finalization.py \
    --ignore=tests/test_efficient_iid_expander.py \
    --ignore=tests/test_guidance_gen_logprobs.py \
    --ignore=tests/test_max_model_len_config.py \
    --ignore=tests/test_model_override_render.py \
    --ignore=tests/test_online_gear.py \
    --ignore=tests/test_tree_update_config.py \
    --ignore=tests/test_tree_update_modes.py \
    || fail "root CPU tests failed"

log "3/12 targeted CPU tests (gear_tree recipe)"
python -m pytest verl/recipe/gear_tree/tests/ -q \
    --ignore=verl/recipe/gear_tree/tests/test_trainer_contracts.py \
    --ignore=verl/recipe/gear_tree/tests/test_vendor_parity.py \
    || fail "gear_tree recipe tests failed"

log "4/12 main config invariants (PLAN.md P0.1 / P0.2 / P0.6)"
python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/gear_tree_trainer.yaml')
cfg = yaml.safe_load(p.read_text())
gt = cfg['gear_tree']
tp = cfg['tree_policy']
arr = cfg['actor_rollout_ref']
rb = gt.get('replay_buffer') or {}
# PLAN.md P0.1: main config picks the segment-average loss with token mean.
assert tp['policy_aggregation'] == 'global_segment_mean', 'PLAN.md P0.1: policy_aggregation must be global_segment_mean'
assert tp['segment_token_reduction'] == 'mean', 'PLAN.md P0.1: default segment_token_reduction must be mean'
assert tp['strict_group_integrity'] is True, 'PLAN.md P0.6: strict_group_integrity must be true'
assert gt['gear']['pilot_execution_mode'] == 'fresh_iid', 'PLAN.md P0.6: pilot_execution_mode must be fresh_iid'
assert gt['gear']['bound_form'] == 'linear', 'PLAN.md P0.6: linear bound required'
assert gt['gear'].get('tail_mode', 'none') == 'none', 'PLAN.md P0.6: tail_mode must be none'
assert float(gt['gear'].get('eps_tail', 0.0)) == 0.0, 'PLAN.md P0.6: eps_tail must be 0'
assert gt['gear']['allocation_runtime'] == 'online_timeout', 'PLAN.md P0.6: allocation_runtime must be online_timeout'
assert gt['gear']['allocation_scope'] == 'per_queue_flush_within_tree', 'PLAN.md P0.6: allocation_scope must be per_queue_flush_within_tree'
assert arr['actor']['policy_loss']['loss_mode'] == 'vdra_segment_mean_ppo', 'PLAN.md P0.1: loss_mode must be vdra_segment_mean_ppo'
assert arr['actor']['policy_loss']['segment_token_reduction'] == 'mean', 'PLAN.md P0.1: actor segment_token_reduction must be mean'
# PLAN.md P0.2: canonical replay names + auto per-question cap.
assert int(rb.get('target_edges_per_iteration', 0)) == 512, 'PLAN.md P0.2: target_edges_per_iteration must be 512'
assert int(rb.get('max_edge_age_iterations', 0)) == 8, 'PLAN.md P0.2: max_edge_age_iterations must be 8'
assert str(rb.get('max_edges_per_question_per_iteration')).lower() == 'auto', 'PLAN.md P0.2: max_edges_per_question_per_iteration must be auto'
assert str(rb.get('replay_sampling_unit', 'edge')) == 'edge', 'PLAN.md P0.2: replay_sampling_unit must be edge'
# PLAN.md P0.3: ppo_mini_batch_size 128, ppo_epochs 1 (so 512/128 = 4).
assert int(arr['actor'].get('ppo_mini_batch_size', 0)) == 128, 'PLAN.md P0.3: ppo_mini_batch_size must be 128'
assert int(arr['actor'].get('ppo_epochs', 1)) == 1, 'PLAN.md P0.3: ppo_epochs must be 1'
# PLAN.md P0.6: scorer topology must resolve one valid mode.
gc = gt['gear']
same = bool(gc.get('scorer_uses_rollout_server', False))
if same:
    assert not gc.get('rollout_api_base'), 'PLAN.md P0.6: rollout_api_base must be null in same-server mode'
else:
    assert gc.get('rollout_api_base'), 'PLAN.md P0.6: rollout_api_base required in two-endpoint mode'
    assert gc.get('scorer_api_base'), 'PLAN.md P0.6: scorer_api_base required in two-endpoint mode'
" || fail "main config invariants failed"

log "4b/12 segment_token_reduction=sum override composes cleanly"
python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/gear_tree_trainer.yaml')
cfg = yaml.safe_load(p.read_text())
# Simulate an ablation override; the fields that must NOT change are the
# allocation, replay, and outer aggregation settings.
tp = cfg['tree_policy']
tp['segment_token_reduction'] = 'sum'
cfg['actor_rollout_ref']['actor']['policy_loss']['segment_token_reduction'] = 'sum'
assert tp['policy_aggregation'] == 'global_segment_mean'
assert cfg['actor_rollout_ref']['actor']['policy_loss']['loss_mode'] == 'vdra_segment_mean_ppo'
assert cfg['gear_tree']['gear']['allocation_runtime'] == 'online_timeout'
assert cfg['gear_tree']['gear']['allocation_scope'] == 'per_queue_flush_within_tree'
assert int(cfg['gear_tree']['replay_buffer']['target_edges_per_iteration']) == 512
assert int(cfg['gear_tree']['replay_buffer']['max_edge_age_iterations']) == 8
" || fail "segment_token_reduction=sum override failed"

log "5/12 real dataclass instantiation (PolicyLossConfig / ActorConfig)"
python -c "
import yaml
from pathlib import Path
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
p = Path('verl/recipe/gear_tree/config/gear_tree_trainer.yaml')
cfg = yaml.safe_load(p.read_text())
policy_loss_cfg = cfg['actor_rollout_ref']['actor']['policy_loss']
# Instantiate the real dataclass to exercise __post_init__ validation.
pl = PolicyLossConfig(
    loss_mode=policy_loss_cfg['loss_mode'],
    segment_token_reduction=policy_loss_cfg['segment_token_reduction'],
)
assert pl.segment_token_reduction == 'mean'
assert pl.loss_mode == 'vdra_segment_mean_ppo'
# Force an invalid override — must raise ValueError.
try:
    PolicyLossConfig(segment_token_reduction='average')
    raise AssertionError('PolicyLossConfig accepted an invalid segment_token_reduction')
except ValueError:
    pass
# Sum override must survive dataclass validation.
pl_sum = PolicyLossConfig(
    loss_mode='vdra_segment_mean_ppo', segment_token_reduction='sum'
)
assert pl_sum.segment_token_reduction == 'sum'
" || fail "PolicyLossConfig / ActorConfig instantiation failed"

log "6/12 smoke A-D configs load"
for overlay in smoke_a_spo_baseline smoke_b_vdra_alloc_legacy_loss smoke_c_uniform_alloc_node_balanced smoke_d_full_vdra; do
    python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/${overlay}.yaml')
cfg = yaml.safe_load(p.read_text())
assert isinstance(cfg, dict) and cfg, 'overlay ${overlay} must load to a non-empty dict'
" || fail "smoke overlay ${overlay} did not load"
done

log "7/12 PLAN.md P0.1 config wiring tests"
python -m pytest verl/recipe/gear_tree/tests/test_policy_loss_config_wiring.py -q \
    || fail "config wiring tests failed"

log "8/12 PLAN.md P0.2 replay auto-cap tests (666 → 33, 888 → 73)"
python -m pytest verl/recipe/gear_tree/tests/test_replay_auto_cap.py -q \
    || fail "replay auto-cap tests failed"

log "9/12 PLAN.md P0.3 optimizer-step accounting tests (512/128 = 4 steps)"
python -m pytest verl/recipe/gear_tree/tests/test_optimizer_step_accounting.py -q \
    || fail "optimizer-step accounting tests failed"

log "10/12 PLAN.md P0.4 batch-slot N_B normalization + microbatch parity"
python -m pytest verl/recipe/gear_tree/tests/test_batch_slot_normalization.py -q \
    || fail "batch-slot normalization tests failed"

log "11/12 PLAN.md P0.5 / P0.6 zero-sparsity + distributed grad + scorer"
python -m pytest verl/recipe/gear_tree/tests/test_zero_advantage_sparsity.py \
    verl/recipe/gear_tree/tests/test_distributed_grad_scaling.py \
    verl/recipe/gear_tree/tests/test_rollout_scorer_verification.py -q \
    || fail "P0.5 / P0.6 tests failed"

log "12a/12 unique tree-ID + transactional replay test"
python -m pytest verl/recipe/gear_tree/tests/test_replay_buffer.py -q \
    -k "distinct_ids or add_raises_on_duplicate or coexist_in_replay or json_roundtrip or transactional_add or intra_batch_duplicate" \
    || fail "unique tree-ID / transactional replay checks failed"

log "12b/12 segment-average reference tests (mean + sum)"
python -m pytest verl/recipe/gear_tree/tests/test_segment_mean_loss.py -q \
    || fail "segment-average reference tests failed"

log "12c/12 segment-count invariants (pre-filter counting, queue identity)"
python -m pytest verl/recipe/gear_tree/tests/test_segment_counts.py -q \
    || fail "segment-count invariants failed"

log "12d/12 full-vs-split gradient parity (legacy ablation)"
python -m pytest verl/recipe/gear_tree/tests/test_vdra_full_vs_split_parity.py -q \
    || fail "full-vs-split gradient parity failed"

log "12e/12 complete-tree replay + reservation-does-not-split-parent"
python -m pytest verl/recipe/gear_tree/tests/test_complete_tree_replay.py -q \
    || fail "complete-tree replay tests failed"

log "12f/12 manifest synthetic lifecycle test (mean + sum)"
python -m pytest verl/recipe/gear_tree/tests/test_manifest_observed_facts.py \
    verl/recipe/gear_tree/tests/test_manifest_wiring.py \
    verl/recipe/gear_tree/tests/test_run_manifest.py -q \
    || fail "manifest synthetic lifecycle failed"

# All checks passed — emit the single acceptance line.
echo "PRE_GPU_CHECK=PASS"
