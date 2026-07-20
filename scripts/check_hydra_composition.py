"""PLAN.md P0.K: REAL Hydra composition of the canonical main config.

Replaces the old ``yaml.safe_load`` shape checks in the pre-GPU gate.
Composes ``gear_tree_trainer.yaml`` through actual Hydra (including the
``pkg://verl.trainer.config`` searchpath and the ``ppo_trainer`` defaults
base), asserts every canonical invariant on the COMPOSED config, composes
the ``segment_token_reduction=sum`` override the same way, and instantiates
the complete typed ``ActorConfig`` (not only ``PolicyLossConfig``) from the
composed actor block.

Exit code 0 and the final ``HYDRA_COMPOSITION=PASS`` line mean success.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "verl" / "recipe" / "gear_tree" / "config"
# PLAN.md M5: allow standalone execution (pre_gpu_check.sh also exports
# PYTHONPATH; this keeps `python scripts/check_hydra_composition.py` working
# without it).
if str(REPO / "verl") not in sys.path:
    sys.path.insert(0, str(REPO / "verl"))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Same compatibility shim the recipe tests use: newer transformers releases
# removed AutoModelForVision2Seq, which verl.utils.model imports eagerly.
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object


def _fail(message: str) -> None:
    print(f"HYDRA_COMPOSITION=FAIL: {message}")
    sys.exit(1)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def compose_main(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="gear_tree_trainer", overrides=overrides or [])


def check_canonical(cfg) -> None:
    tree_policy = cfg.tree_policy
    _require(
        tree_policy.policy_aggregation == "segment_mean",
        f"tree_policy.policy_aggregation={tree_policy.policy_aggregation}",
    )
    _require(
        tree_policy.segment_token_reduction == "mean",
        f"tree_policy.segment_token_reduction={tree_policy.segment_token_reduction}",
    )
    _require(
        bool(tree_policy.strict_group_integrity),
        "tree_policy.strict_group_integrity must be true",
    )

    gear = cfg.gear_tree.gear
    _require(
        gear.pilot_execution_mode == "fresh_iid",
        f"gear.pilot_execution_mode={gear.pilot_execution_mode}",
    )
    _require(
        gear.allocation_runtime == "online_timeout",
        f"gear.allocation_runtime={gear.allocation_runtime}",
    )
    # PLAN.md M5: the strict canonical triple requires tree_update_mode=spo.
    _require(
        str(cfg.gear_tree.tree_update_mode) == "spo",
        f"gear_tree.tree_update_mode={cfg.gear_tree.tree_update_mode}",
    )
    _require(
        bool(gear.get("strict_vdra", True)),
        "gear.strict_vdra must be true on the canonical main path",
    )

    actor = cfg.actor_rollout_ref.actor
    _require(
        actor.policy_loss.loss_mode == "vdra_segment_mean_ppo",
        f"actor.policy_loss.loss_mode={actor.policy_loss.loss_mode}",
    )
    _require(
        actor.policy_loss.segment_token_reduction == "mean",
        f"actor.policy_loss.segment_token_reduction="
        f"{actor.policy_loss.segment_token_reduction}",
    )
    # Canonical normalization is the tree segment mean w_s = 1/(N_T*N_seg(T));
    # PLAN.md §1.3: the actor-level aggregation must advertise the canonical
    # paper objective and agree with tree_policy.policy_aggregation.
    _require(
        str(actor.policy_loss.get("policy_aggregation", "")) == "segment_mean",
        "actor.policy_loss.policy_aggregation must be 'segment_mean' on the "
        "canonical main path",
    )
    _require(int(actor.ppo_mini_batch_size) == 128, "ppo_mini_batch_size != 128")
    _require(int(actor.ppo_epochs) == 1, "ppo_epochs != 1")

    replay = cfg.gear_tree.replay_buffer
    _require(
        int(replay.target_edges_per_iteration) == 512,
        f"target_edges_per_iteration={replay.target_edges_per_iteration}",
    )
    _require(
        int(replay.max_edge_age_iterations) == 8,
        f"max_edge_age_iterations={replay.max_edge_age_iterations}",
    )
    _require(
        str(replay.max_edges_per_question_per_iteration) == "auto",
        "max_edges_per_question_per_iteration != auto",
    )
    _require(
        replay.replay_sampling_unit == "edge",
        f"replay_sampling_unit={replay.replay_sampling_unit}",
    )
    _require(
        replay.underfilled_update_policy == "postpone_until_divisible",
        f"underfilled_update_policy={replay.underfilled_update_policy}",
    )
    _require(
        int(replay.target_edges_per_iteration)
        % int(actor.ppo_mini_batch_size)
        == 0,
        "target not divisible by ppo_mini_batch_size",
    )


# PLAN.md M5: canonical VDRA policy-loss fields. Finding one of these at the
# actor TOP level (outside actor.policy_loss) while the dataclass schema does
# not declare it there means a canonical field was misplaced — that is a hard
# FAIL, never a silent drop.
CANONICAL_POLICY_LOSS_FIELDS = {
    "loss_mode",
    "segment_token_reduction",
    "use_prob_mask",
    "ratio_threshold",
}


def instantiate_actor_config(cfg) -> None:
    """PLAN.md P0.K/M5: build the COMPLETE typed ActorConfig from the composed
    actor block — a Hydra-composed run must round-trip into the dataclass
    schema, not only into PolicyLossConfig. Unknown fields are never silently
    deleted: a misplaced canonical VDRA field FAILS the gate, and legitimate
    upstream runtime-only fields are excluded with an explicit WARN listing.
    """
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    actor_cfg = cfg.actor_rollout_ref.actor
    container = OmegaConf.to_container(actor_cfg, resolve=True)
    target = getattr(actor_cfg, "_target_", None)
    known = set(ActorConfig.__dataclass_fields__) | set(
        __import__(
            "verl.workers.config.actor", fromlist=["FSDPActorConfig"]
        ).FSDPActorConfig.__dataclass_fields__
    )
    dropped = sorted(
        key for key in container if key not in known and key != "_target_"
    )
    misplaced = sorted(set(dropped) & CANONICAL_POLICY_LOSS_FIELDS)
    if misplaced:
        _fail(
            "canonical VDRA field(s) misplaced at the actor top level "
            f"instead of actor.policy_loss: {misplaced}"
        )
    if dropped:
        # Legitimate upstream runtime-only fields the dataclass schema does
        # not declare. Report them explicitly; never delete silently.
        print(
            "HYDRA_COMPOSITION=WARN: upstream runtime-only actor fields not "
            f"in the dataclass schema (excluded from typed round-trip): {dropped}"
        )
    # The policy_loss block itself must round-trip exactly: any key the
    # typed PolicyLossConfig does not declare is a misplaced canonical field.
    policy_loss_container = dict(container.get("policy_loss") or {})
    unknown_policy_loss = sorted(
        key
        for key in policy_loss_container
        if key not in set(PolicyLossConfig.__dataclass_fields__)
        and key != "_target_"
    )
    if unknown_policy_loss:
        _fail(
            "actor.policy_loss carries field(s) unknown to the typed "
            f"PolicyLossConfig schema: {unknown_policy_loss}"
        )
    cleaned = {
        key: value for key, value in container.items() if key in known
    }
    cleaned.setdefault("strategy", "fsdp")
    cleaned["ppo_micro_batch_size_per_gpu"] = cleaned.get(
        "ppo_micro_batch_size_per_gpu"
    ) or 32
    dc_cfg = OmegaConf.create(cleaned)
    if target:
        dc_cfg["_target_"] = target
        actor = omega_conf_to_dataclass(dc_cfg)
    else:
        from verl.workers.config.actor import FSDPActorConfig

        actor = omega_conf_to_dataclass(dc_cfg, dataclass_type=FSDPActorConfig)
    _require(
        isinstance(actor, ActorConfig),
        f"instantiated actor config has type {type(actor)}",
    )
    _require(
        isinstance(actor.policy_loss, PolicyLossConfig)
        or actor.policy_loss.get("loss_mode", None) is not None,
        "policy_loss block missing from instantiated ActorConfig",
    )
    _require(
        str(actor.policy_loss.loss_mode) == "vdra_segment_mean_ppo",
        f"instantiated loss_mode={actor.policy_loss.loss_mode}",
    )
    print(
        "instantiated ActorConfig:",
        type(actor).__name__,
        "policy_loss.loss_mode=",
        actor.policy_loss.loss_mode,
        "segment_token_reduction=",
        actor.policy_loss.segment_token_reduction,
    )


def run_trainer_cross_level_validation(cfg, label: str) -> None:
    """PLAN.md M5: run the EXACT tree_policy <-> actor.policy_loss validation
    the trainer runs at startup (extracted to config_validation), instead of
    re-implementing it in this gate."""
    from recipe.gear_tree.config_validation import (
        validate_policy_loss_consistency,
    )

    try:
        validate_policy_loss_consistency(cfg)
    except ValueError as exc:
        _fail(f"trainer cross-level validation rejected {label}: {exc}")
    print(f"trainer cross-level validation ({label}): OK")


def main() -> None:
    cfg = compose_main()
    check_canonical(cfg)
    print("composed canonical gear_tree_trainer: OK")

    cfg_sum = compose_main(
        overrides=[
            "tree_policy.segment_token_reduction=sum",
            "actor_rollout_ref.actor.policy_loss.segment_token_reduction=sum",
        ]
    )
    _require(
        cfg_sum.actor_rollout_ref.actor.policy_loss.segment_token_reduction
        == "sum",
        "sum override did not reach actor.policy_loss",
    )
    _require(
        cfg_sum.tree_policy.segment_token_reduction == "sum",
        "sum override did not reach tree_policy",
    )
    print("composed sum override: OK")

    run_trainer_cross_level_validation(cfg, "canonical mean config")
    run_trainer_cross_level_validation(cfg_sum, "sum override config")

    instantiate_actor_config(cfg)
    print("HYDRA_COMPOSITION=PASS")


if __name__ == "__main__":
    main()
