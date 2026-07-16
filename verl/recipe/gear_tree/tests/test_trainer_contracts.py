import inspect
from types import SimpleNamespace

import pytest

import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _trainer(balance_batch=False, target_edges=512, mini_batch=128):
    obj = object.__new__(RayGearTreeTrainer)
    obj.tokenizer = _Tokenizer()
    obj.config = _Cfg(
        data=_Cfg(max_prompt_length=4, max_response_length=3),
        trainer=_Cfg(balance_batch=balance_batch, default_local_dir="/tmp"),
        actor_rollout_ref=_Cfg(actor=_Cfg(ppo_mini_batch_size=mini_batch)),
        gear_tree={
            "replay_buffer": {
                "target_edges_per_update": target_edges,
                "max_edges_per_question": 32,
                "max_edge_age": 8,
                "sampling_seed": 0,
            }
        },
    )
    obj.balance_calls = 0

    def _balance_batch(batch, metrics):
        obj.balance_calls += 1
        metrics["balanced"] = 1

    obj._balance_batch = _balance_batch
    return obj


def _edge(edge_id="e"):
    return {
        "edge_id": edge_id,
        "question_id": "q",
        "query_token_ids": [5, 6],
        "response_token_ids": [7, 8],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": 1.0,
        "value": 0.5,
        "reward": 1.0,
        "depth": 0,
        "leaf": False,
        "pruned": False,
        "tree_update_mode": "spo",
    }


def test_update_batch_sets_required_metadata_and_old_log_probs():
    trainer = _trainer(balance_batch=True)
    metrics = {}
    batch = trainer._edges_to_update_batch([_edge()], metrics)
    assert trainer.balance_calls == 1
    assert metrics["balanced"] == 1
    assert batch.meta_info["multi_turn"] is False
    assert batch.meta_info["global_token_num"] == batch.batch["attention_mask"].sum(dim=-1).tolist()
    assert "old_log_probs" in batch.batch
    assert batch.batch["old_log_probs"].shape == batch.batch["responses"].shape


def test_replay_startup_validates_mini_batch_divisibility():
    trainer = _trainer(target_edges=510, mini_batch=128)
    with pytest.raises(ValueError, match="target_edges_per_update"):
        trainer._validate_replay_startup()


def test_generated_edges_require_stored_generation_logprobs():
    trainer = _trainer()
    edge = _edge()
    edge.pop("actor_shifted_log_probs")
    with pytest.raises(ValueError, match="generation-time"):
        trainer._normalize_generated_edges([edge], snapshot_id="global_step:1")


def test_trainer_source_does_not_recompute_old_log_probs_in_fit():
    source = inspect.getsource(RayGearTreeTrainer.fit)
    assert "compute_log_prob" not in source
