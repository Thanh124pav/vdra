from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer


def _edge(edge_id, question_id="q", step=0, depth=0, lp=-0.1, advantage=1.0):
    return {
        "edge_id": str(edge_id),
        "question_id": str(question_id),
        "generation_step": int(step),
        "policy_snapshot_id": "snap",
        "query_token_ids": [1, 2],
        "response_token_ids": [3, 4],
        "actor_shifted_log_probs": [lp, lp - 0.1],
        "advantage": advantage,
        "value": 0.5,
        "reward": 1.0,
        "depth": depth,
        "leaf": False,
        "pruned": False,
        "tree_update_mode": "spo",
        "tree_update_local_advantage": advantage,
        "tree_update_global_advantage": advantage,
        "tree_update_parent_reward": 0.0,
        "tree_update_child_reward": 1.0,
        "tree_update_root_reward": 1.0,
    }


def _buffer(**kwargs):
    cfg = {
        "target_edges_per_update": 512,
        "max_edges_per_question": 32,
        "max_edge_age": 8,
        "sampling_seed": 13,
    }
    cfg.update(kwargs)
    return GearTreeReplayBuffer(**cfg)


def test_per_question_cap_applies_before_global_cap():
    buf = _buffer(target_edges_per_update=512, max_edges_per_question=32)
    buf.add([_edge(i, question_id="one") for i in range(100)], generation_step=0, policy_snapshot_id="snap")
    sampled, stats = buf.sample_for_update(current_step=1)
    assert len(sampled) == 32
    assert stats["buffer/edges_per_question_max"] == 32


def test_global_cap_removes_only_sampled_edges():
    buf = _buffer(target_edges_per_update=512, max_edges_per_question=1000)
    buf.add([_edge(i, question_id=i // 100) for i in range(1000)], generation_step=0, policy_snapshot_id="snap")
    sampled, stats = buf.sample_for_update(current_step=1)
    assert len(sampled) == 512
    assert len(buf) == 488
    assert stats["removed_edge_ids"] == sorted(edge["edge_id"] for edge in sampled)


def test_underfill_uses_available_without_duplication():
    buf = _buffer(target_edges_per_update=512, max_edges_per_question=512)
    buf.add([_edge(i, question_id=i) for i in range(300)], generation_step=0, policy_snapshot_id="snap")
    sampled, stats = buf.sample_for_update(current_step=1)
    assert len(sampled) == 300
    assert len({edge["edge_id"] for edge in sampled}) == 300
    assert stats["buffer/underfilled"] == 1.0
    assert len(buf) == 0


def test_age_expiration_boundary():
    buf = _buffer(max_edge_age=8, target_edges_per_update=10)
    buf.add([_edge("survive", step=3), _edge("expire", step=2)], generation_step=0, policy_snapshot_id="snap")
    sampled, stats = buf.sample_for_update(current_step=10)
    assert [edge["edge_id"] for edge in sampled] == ["survive"]
    assert stats["buffer/expired_edges"] == 1


def test_deterministic_sampling_for_same_seed_and_step():
    edges = [_edge(i, question_id=i // 50) for i in range(600)]
    left = _buffer(sampling_seed=7, max_edges_per_question=100)
    right = _buffer(sampling_seed=7, max_edges_per_question=100)
    left.add(edges, generation_step=0, policy_snapshot_id="snap")
    right.add(edges, generation_step=0, policy_snapshot_id="snap")
    l_sampled, _ = left.sample_for_update(current_step=4)
    r_sampled, _ = right.sample_for_update(current_step=4)
    assert [edge["edge_id"] for edge in l_sampled] == [edge["edge_id"] for edge in r_sampled]


def test_different_steps_can_change_sample():
    edges = [_edge(i, question_id=i // 50) for i in range(600)]
    left = _buffer(sampling_seed=7, max_edges_per_question=100)
    right = _buffer(sampling_seed=7, max_edges_per_question=100)
    left.add(edges, generation_step=0, policy_snapshot_id="snap")
    right.add(edges, generation_step=0, policy_snapshot_id="snap")
    l_sampled, _ = left.sample_for_update(current_step=4)
    r_sampled, _ = right.sample_for_update(current_step=5)
    assert [edge["edge_id"] for edge in l_sampled] != [edge["edge_id"] for edge in r_sampled]


def test_checkpoint_round_trip_preserves_edges_and_values(tmp_path):
    buf = _buffer(target_edges_per_update=10)
    edges = [_edge("a", lp=-0.123, advantage=2.5), _edge("b", lp=-0.456, advantage=-1.5)]
    buf.add(edges, generation_step=2, policy_snapshot_id="snap")
    buf.save(tmp_path)
    restored = GearTreeReplayBuffer.load(tmp_path)
    assert restored.edges() == buf.edges()
    sampled, _ = restored.sample_for_update(current_step=3)
    assert sampled[0]["actor_shifted_log_probs"] == [-0.123, -0.223]
    assert sampled[0]["advantage"] == 2.5


def test_same_sampler_is_method_independent():
    buf = _buffer(target_edges_per_update=4, max_edges_per_question=4)
    spo = [_edge(f"spo-{i}", question_id="q", advantage=1.0) for i in range(4)]
    vdra = [_edge(f"vdra-{i}", question_id="q", advantage=1.0) for i in range(4)]
    buf.add(spo + vdra, generation_step=0, policy_snapshot_id="snap")
    sampled, _ = buf.sample_for_update(current_step=1)
    assert len(sampled) == 4
    assert all(edge["tree_update_mode"] == "spo" for edge in sampled)


def test_peek_sampling_does_not_remove_until_explicit_remove():
    buf = _buffer(target_edges_per_update=4, max_edges_per_question=10)
    buf.add([_edge(i, question_id=i) for i in range(4)], generation_step=0, policy_snapshot_id="snap")
    sampled, stats = buf.sample_for_update(current_step=1, remove=False)
    assert len(sampled) == 4
    assert len(buf) == 4
    assert buf.remove(stats["removed_edge_ids"]) == stats["removed_edge_ids"]
    assert len(buf) == 0


def test_reservation_hides_edges_until_commit_or_rollback():
    buf = _buffer(target_edges_per_update=2, max_edges_per_question=10)
    buf.add([_edge(i, question_id=i) for i in range(3)], generation_step=0, policy_snapshot_id="snap")
    reservation = buf.reserve_for_update(current_step=1)
    assert len(reservation.edges) == 2
    assert len(buf) == 3

    sampled, stats = buf.sample_for_update(current_step=1, remove=False)
    assert {edge["edge_id"] for edge in sampled}.isdisjoint(reservation.edge_ids)
    assert len(sampled) == 1

    buf.rollback(reservation)
    assert len(buf) == 3
    sampled_after_rollback, _ = buf.sample_for_update(current_step=1, remove=False)
    assert set(reservation.edge_ids).issubset({edge["edge_id"] for edge in sampled_after_rollback})


def test_commit_removes_only_reserved_edges():
    buf = _buffer(target_edges_per_update=2, max_edges_per_question=10)
    buf.add([_edge(i, question_id=i) for i in range(3)], generation_step=0, policy_snapshot_id="snap")
    reservation = buf.reserve_for_update(current_step=1)
    removed = buf.commit(reservation)
    assert tuple(removed) == reservation.edge_ids
    assert len(buf) == 1
    assert {edge["edge_id"] for edge in buf.edges()}.isdisjoint(reservation.edge_ids)


def test_add_raises_on_duplicate_edge_id():
    # PLAN.md P0.2: duplicate edge_ids indicate the rollout did not stamp a
    # unique tree_instance_id. ReplayBuffer.add must raise, never silently
    # overwrite.
    import pytest

    buf = _buffer()
    buf.add([_edge("dup")], generation_step=0, policy_snapshot_id="snap")
    with pytest.raises(ValueError, match="tree_instance_id"):
        buf.add([_edge("dup", advantage=2.0)], generation_step=0, policy_snapshot_id="snap")


def test_two_trees_for_same_question_and_snapshot_have_distinct_ids():
    # PLAN.md P0.2 acceptance: two independent trees for the same
    # (question, snapshot, iteration) tuple must not collapse to one id.
    from recipe.gear_tree.tree_rollout import make_tree_instance_id

    a = make_tree_instance_id(
        policy_snapshot_id="snap",
        rollout_iteration=1,
        stable_question_id="q42",
    )
    b = make_tree_instance_id(
        policy_snapshot_id="snap",
        rollout_iteration=1,
        stable_question_id="q42",
    )
    assert a != b


def test_tree_instance_id_survives_json_roundtrip():
    import json

    from recipe.gear_tree.tree_rollout import make_tree_instance_id

    tid = make_tree_instance_id(
        policy_snapshot_id="global_step:7",
        rollout_iteration=3,
        stable_question_id="q99",
        tree_instance_uuid="abc123",
    )
    dumped = json.dumps({"tree_id": tid})
    loaded = json.loads(dumped)
    assert loaded["tree_id"] == tid


def test_two_trees_edges_coexist_in_replay():
    # PLAN.md P0.2 acceptance: their edges coexist in replay without collision.
    from recipe.gear_tree.tree_rollout import make_tree_instance_id
    import hashlib

    buf = _buffer()
    edges = []
    for _ in range(2):
        tid = make_tree_instance_id(
            policy_snapshot_id="snap",
            rollout_iteration=0,
            stable_question_id="q42",
        )
        digest = hashlib.blake2b(tid.encode("utf-8"), digest_size=8).hexdigest()
        edges.append(_edge(f"e:{digest}", question_id="q42"))
    buf.add(edges, generation_step=0, policy_snapshot_id="snap")
    assert len(buf) == 2
