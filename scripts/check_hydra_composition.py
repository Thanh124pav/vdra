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

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "verl" / "recipe" / "gear_tree" / "config"


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
        tree_policy.policy_aggregation == "global_segment_mean",
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


def instantiate_actor_config(cfg) -> None:
    """PLAN.md P0.K: build the COMPLETE typed ActorConfig from the composed
    actor block — a Hydra-composed run must round-trip into the dataclass
    schema, not only into PolicyLossConfig."""
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    actor_cfg = cfg.actor_rollout_ref.actor
    container = OmegaConf.to_container(actor_cfg, resolve=True)
    # The composed block carries runtime-resolved fields the dataclass does
    # not declare; keep only declared fields, exactly like the worker does
    # through its config plumbing.
    target = getattr(actor_cfg, "_target_", None)
    known = set(ActorConfig.__dataclass_fields__) | set(
        __import__(
            "verl.workers.config.actor", fromlist=["FSDPActorConfig"]
        ).FSDPActorConfig.__dataclass_fields__
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

    instantiate_actor_config(cfg)
    print("HYDRA_COMPOSITION=PASS")


if __name__ == "__main__":
    main()
