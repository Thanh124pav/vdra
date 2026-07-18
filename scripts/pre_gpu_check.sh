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

log "1/8 compileall vdra_core and gear_tree"
python -m compileall -q vdra_core verl/recipe/gear_tree \
    || fail "compileall failed"

log "2/8 targeted CPU tests (root)"
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

log "3/8 targeted CPU tests (gear_tree recipe)"
python -m pytest verl/recipe/gear_tree/tests/ -q \
    --ignore=verl/recipe/gear_tree/tests/test_trainer_contracts.py \
    --ignore=verl/recipe/gear_tree/tests/test_vendor_parity.py \
    || fail "gear_tree recipe tests failed"

log "4/8 main config invariants"
python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/gear_tree_trainer.yaml')
cfg = yaml.safe_load(p.read_text())
gt = cfg['gear_tree']
tp = cfg['tree_policy']
arr = cfg['actor_rollout_ref']
assert gt['only_adv_greater_than_zero'] is False, 'PLAN.md P0.1: only_adv_greater_than_zero must be false'
assert tp['policy_aggregation'] == 'vdra_node_balanced', 'PLAN.md P0.4: policy_aggregation must be vdra_node_balanced'
assert tp['strict_group_integrity'] is True, 'PLAN.md P0.6: strict_group_integrity must be true'
assert gt['gear']['pilot_execution_mode'] == 'fresh_iid', 'PLAN.md P0.1: pilot_execution_mode must be fresh_iid'
assert gt['gear']['bound_form'] == 'linear', 'PLAN.md linear bound required'
assert arr['actor']['policy_loss']['loss_mode'] == 'vdra_node_balanced_ppo', 'PLAN.md P0.4: loss_mode must be vdra_node_balanced_ppo'
" || fail "main config invariants failed"

log "5/8 smoke A-D configs load"
for overlay in smoke_a_spo_baseline smoke_b_vdra_alloc_legacy_loss smoke_c_uniform_alloc_node_balanced smoke_d_full_vdra; do
    python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/${overlay}.yaml')
cfg = yaml.safe_load(p.read_text())
assert isinstance(cfg, dict) and cfg, 'overlay ${overlay} must load to a non-empty dict'
" || fail "smoke overlay ${overlay} did not load"
done

log "6/8 unique tree-ID test"
python -m pytest verl/recipe/gear_tree/tests/test_replay_buffer.py -q -k "distinct_ids or add_raises_on_duplicate or coexist_in_replay or json_roundtrip" \
    || fail "unique tree-ID checks failed"

log "7/8 full-vs-split gradient parity test"
python -m pytest verl/recipe/gear_tree/tests/test_vdra_full_vs_split_parity.py -q \
    || fail "full-vs-split gradient parity failed"

log "8/8 manifest synthetic lifecycle test"
python -m pytest verl/recipe/gear_tree/tests/test_manifest_observed_facts.py -q \
    || fail "manifest synthetic lifecycle failed"

# All checks passed — emit the single acceptance line.
echo "PRE_GPU_CHECK=PASS"
