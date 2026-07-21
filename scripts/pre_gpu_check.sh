#!/usr/bin/env bash
# PLAN.md P0.K — one pre-GPU gate command.
#
# Runs every CPU-verifiable invariant that must hold before launching a GPU
# smoke run — including the ACTUAL production wiring (real Hydra
# composition, real edge reservation dispatch, real
# DataParallelPPOActor.update_policy control flow, real two-process
# distributed gradient parity, and the CPU trainer contracts) — and prints
# exactly one line on success:
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
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

log() {
    printf "[pre_gpu_check] %s\n" "$*" >&2
}

fail() {
    printf "[pre_gpu_check] FAIL: %s\n" "$*" >&2
    exit 1
}

log "1/20 compileall vdra_core and gear_tree"
python -m compileall -q vdra_core verl/recipe/gear_tree \
    || fail "compileall failed"

log "1b/20 real Ray entrypoint imports"
python - <<'PY'
import verl.trainer.main_ppo
import recipe.gear_tree.main_gear_tree
print("RAY_ENTRYPOINT_IMPORT=PASS")
PY

log "1c/20 Ray entrypoint actor healthchecks"
python -m pytest verl/recipe/gear_tree/tests/test_main_gear_tree_entrypoint.py -q \
    || fail "Ray entrypoint actor tests failed"

log "2/20 targeted CPU tests (root)"
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

log "3/20 targeted CPU tests (gear_tree recipe, trainer contracts INCLUDED)"
# PLAN.md P0.K: test_trainer_contracts.py is no longer ignored — it runs on
# CPU with torchdata/peft/datasets installed. Only vendor parity (full
# GPU/vLLM stack) stays out of the CPU gate.
python -m pytest verl/recipe/gear_tree/tests/ -q \
    --ignore=verl/recipe/gear_tree/tests/test_vendor_parity.py \
    || fail "gear_tree recipe tests failed"

log "4/20 REAL Hydra composition (main config + sum override + full ActorConfig)"
# PLAN.md P0.K: actual hydra.compose through the pkg://verl.trainer.config
# searchpath — NOT yaml.safe_load — plus complete typed ActorConfig
# instantiation from the composed actor block.
python scripts/check_hydra_composition.py \
    || fail "real Hydra composition failed"

log "5/20 PolicyLossConfig schema validation (invalid value must raise)"
python -c "
from verl.workers.config.actor import PolicyLossConfig
pl = PolicyLossConfig(loss_mode='vdra_segment_mean_ppo', segment_token_reduction='mean')
assert pl.segment_token_reduction == 'mean'
try:
    PolicyLossConfig(segment_token_reduction='average')
    raise AssertionError('PolicyLossConfig accepted an invalid segment_token_reduction')
except ValueError:
    pass
pl_sum = PolicyLossConfig(loss_mode='vdra_segment_mean_ppo', segment_token_reduction='sum')
assert pl_sum.segment_token_reduction == 'sum'
" || fail "PolicyLossConfig schema validation failed"

log "6/20 smoke A-D overlay files parse"
for overlay in smoke_a_spo_baseline smoke_b_vdra_alloc_legacy_loss smoke_c_uniform_alloc_node_balanced smoke_d_full_vdra; do
    python -c "
import yaml
from pathlib import Path
p = Path('verl/recipe/gear_tree/config/${overlay}.yaml')
cfg = yaml.safe_load(p.read_text())
assert isinstance(cfg, dict) and cfg, 'overlay ${overlay} must load to a non-empty dict'
" || fail "smoke overlay ${overlay} did not load"
done

# ---------------------------------------------------------------------------
# Production wiring (PLAN.md P0.A-P0.J) — actual production functions, not
# synthetic mirrors.
# ---------------------------------------------------------------------------

log "7/20 P0.A strict main dispatches to EDGE reservation (cap 33/73, 516→512)"
python -m pytest verl/recipe/gear_tree/tests/test_trainer_replay_dispatch.py -q \
    || fail "edge reservation dispatch tests failed"

log "8/20 P0.B construction vs replay-batch validator split"
python -m pytest verl/recipe/gear_tree/tests/test_construction_vs_replay_validation.py -q \
    || fail "construction/replay validator tests failed"

log "9/20 P0.C canonical edges_to_dataproto carries NO objective weights"
python -m pytest verl/recipe/gear_tree/tests/test_canonical_dataproto_no_objective_weights.py -q \
    || fail "canonical DataProto weight tests failed"

log "10/20 P0.D exact sample size + 128-divisibility (no tail batch)"
python -m pytest verl/recipe/gear_tree/tests/test_trainer_batch_cardinality.py -q \
    || fail "batch cardinality tests failed"

log "11/20 P0.E checkpoint/resume restores rollout_iteration (no negative ages)"
python -m pytest verl/recipe/gear_tree/tests/test_trainer_state_checkpoint.py -q \
    || fail "trainer state checkpoint tests failed"

log "12/20 P0.E crossed-threshold save/eval triggers + unambiguous logging"
python -m pytest verl/recipe/gear_tree/tests/test_threshold_crossing_and_logging.py -q \
    || fail "threshold crossing tests failed"

log "13/20 P0.G zero-advantage dense production path (all-zero batch, exact advantage)"
python -m pytest verl/recipe/gear_tree/tests/test_zero_adv_production_skip.py -q \
    || fail "zero-advantage dense-path tests failed"

log "14/20 P0.H strict tree/edge identity + collision detection"
python -m pytest verl/recipe/gear_tree/tests/test_strict_tree_identity.py -q \
    || fail "strict identity tests failed"

log "15/20 P0.K REAL DataParallelPPOActor.update_policy (512/128 = 4 real steps)"
python -m pytest verl/recipe/gear_tree/tests/test_actor_update_control_flow.py -q \
    || fail "actor update control-flow tests failed"

log "16/20 P0.I REAL two-process torch.distributed gradient parity (gloo/DDP)"
python -m pytest verl/recipe/gear_tree/tests/test_distributed_grad_parity.py -q \
    || fail "two-process distributed parity failed"

# ---------------------------------------------------------------------------
# Per-phase regression suites retained from earlier plan items.
# ---------------------------------------------------------------------------

log "17/20 P0.1/P0.F config wiring + P0.2 auto-cap + P0.3 step accounting"
python -m pytest verl/recipe/gear_tree/tests/test_policy_loss_config_wiring.py \
    verl/recipe/gear_tree/tests/test_replay_auto_cap.py \
    verl/recipe/gear_tree/tests/test_optimizer_step_accounting.py -q \
    || fail "config wiring / auto-cap / step accounting tests failed"

log "18/20 P0.4 batch-slot N_B + zero-sparsity identity + scorer + replay buffer"
python -m pytest verl/recipe/gear_tree/tests/test_batch_slot_normalization.py \
    verl/recipe/gear_tree/tests/test_zero_advantage_sparsity.py \
    verl/recipe/gear_tree/tests/test_distributed_grad_scaling.py \
    verl/recipe/gear_tree/tests/test_rollout_scorer_verification.py \
    verl/recipe/gear_tree/tests/test_replay_buffer.py -q \
    || fail "batch-slot / sparsity / scorer / replay buffer tests failed"

log "19/20 segment loss + segment counts + manifest observed facts"
python -m pytest verl/recipe/gear_tree/tests/test_segment_mean_loss.py \
    verl/recipe/gear_tree/tests/test_segment_counts.py \
    verl/recipe/gear_tree/tests/test_manifest_observed_facts.py \
    verl/recipe/gear_tree/tests/test_manifest_wiring.py \
    verl/recipe/gear_tree/tests/test_run_manifest.py -q \
    || fail "segment loss / counts / manifest tests failed"

# ---------------------------------------------------------------------------
# Ablation (non-canonical). These are NOT canonical acceptance criteria —
# complete-tree replay exists only behind replay_sampling_unit=complete_tree.
# ---------------------------------------------------------------------------

log "20/20 ablation (non-canonical): complete-tree replay + node-balanced parity"
python -m pytest verl/recipe/gear_tree/tests/test_complete_tree_replay.py \
    verl/recipe/gear_tree/tests/test_vdra_full_vs_split_parity.py -q \
    || fail "ablation tests failed"

# All checks passed — emit the single acceptance line.
echo "PRE_GPU_CHECK=PASS"
