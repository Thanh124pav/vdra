"""PLAN.md P0.H: strict tree/edge identity.

* Strict extraction refuses the ambiguous ``snapshot:question`` fallback.
* Strict normalization requires make_tree_instance_id-derived identities
  and derives edge IDs from (tree identity | parent group | child segment).
* Manifest verification detects real collisions among multiple stochastic
  trees for one (question, snapshot) — stronger than ``bool(set(ids))``.
"""

from __future__ import annotations

import pytest

from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.tree_data import (
    derive_edge_id,
    normalize_generated_edges,
    verify_tree_instance_id_uniqueness,
)
from recipe.gear_tree.tree_rollout import make_tree_instance_id


def _tree_without_instance_id():
    return {
        "text": "root",
        "full_text": "root",
        "full_token_ids": [1, 2],
        "token_ids": [1, 2],
        "reward": 0.0,
        "gear_segment_id": "seg-root",
        "_request_object": {"_treetune__idx": 0},
        "policy_snapshot_id": "snap0",
        "children": [
            {
                "text": "c0",
                "token_ids": [10],
                "response_token_ids": [10],
                "actor_shifted_log_probs": [-0.1],
                "reward": 1.0,
                "children": [],
                "gear_segment_id": "seg-c0",
            }
        ],
    }


class TestMakeTreeInstanceId:
    def test_same_question_snapshot_iteration_gives_distinct_ids(self):
        a = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=3,
            stable_question_id="q0",
        )
        b = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=3,
            stable_question_id="q0",
        )
        assert a != b

    def test_id_embeds_required_components(self):
        tid = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=7,
            stable_question_id="q42",
        )
        assert "snap0" in tid
        assert "iter:7" in tid
        assert "q:q42" in tid


class TestStrictExtractionRefusesFallback:
    def test_strict_raises_without_tree_instance_id(self):
        with pytest.raises(ValueError, match="tree_instance_id"):
            extract_edges_from_tree(
                _tree_without_instance_id(), strict_fresh_iid=True
            )

    def test_non_strict_keeps_legacy_fallback(self):
        edges = extract_edges_from_tree(
            _tree_without_instance_id(), strict_fresh_iid=False
        )
        assert edges
        assert edges[0]["tree_id"] == "snap0:0"

    def test_strict_accepts_stamped_instance_id(self):
        tree = _tree_without_instance_id()
        tree["tree_instance_id"] = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=1,
            stable_question_id="0",
        )
        edges = extract_edges_from_tree(tree, strict_fresh_iid=True)
        assert edges
        assert edges[0]["tree_id"] == tree["tree_instance_id"]

    def test_strict_rejects_legacy_tree_id_only(self):
        # PLAN.md M3: a generic legacy tree_id (e.g. "t0") alone must not
        # satisfy strict identity — it cannot distinguish two stochastic
        # trees for the same question/snapshot/iteration.
        tree = _tree_without_instance_id()
        tree["tree_id"] = "t0"
        with pytest.raises(ValueError, match="tree_instance_id"):
            extract_edges_from_tree(tree, strict_fresh_iid=True)

    def test_extracted_edges_stamp_tree_instance_id(self):
        tree = _tree_without_instance_id()
        tree["tree_instance_id"] = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=1,
            stable_question_id="0",
        )
        edges = extract_edges_from_tree(tree, strict_fresh_iid=True)
        assert edges
        assert edges[0]["tree_instance_id"] == tree["tree_instance_id"]
        assert edges[0]["tree_id"] == edges[0]["tree_instance_id"]


class TestStrictNormalizer:
    def _edge(self, **overrides) -> dict:
        edge = {
            "tree_instance_id": "snap0|iter:1|q:q0|t:0000-abcd",
            "parent_group_id": "pg0",
            "child_segment_id": "c0",
            "question_id": "q0",
            "response_token_ids": [1, 2],
            "actor_shifted_log_probs": [-0.1, -0.2],
        }
        edge.update(overrides)
        return edge

    def test_strict_requires_identity_components(self):
        with pytest.raises(ValueError, match="P0.H"):
            normalize_generated_edges(
                [self._edge(tree_instance_id=None)],
                snapshot_id="snap0",
                strict=True,
            )
        with pytest.raises(ValueError, match="parent_group_id"):
            normalize_generated_edges(
                [self._edge(parent_group_id=None)],
                snapshot_id="snap0",
                strict=True,
            )

    def test_strict_edge_id_derives_from_tree_identity_plus_child(self):
        [a] = normalize_generated_edges(
            [self._edge()], snapshot_id="snap0", strict=True
        )
        [b] = normalize_generated_edges(
            [self._edge(tree_instance_id="snap0|iter:1|q:q0|t:1111-ffff")],
            snapshot_id="snap0",
            strict=True,
        )
        [c] = normalize_generated_edges(
            [self._edge(child_segment_id="c1")],
            snapshot_id="snap0",
            strict=True,
        )
        # Distinct tree identity or child identity => distinct edge_id.
        assert a["edge_id"] != b["edge_id"]
        assert a["edge_id"] != c["edge_id"]
        # Deterministic for identical identity.
        [a2] = normalize_generated_edges(
            [self._edge()], snapshot_id="snap0", strict=True
        )
        assert a["edge_id"] == a2["edge_id"]

    def test_two_trees_same_question_get_distinct_edge_ids(self):
        tid_a = make_tree_instance_id(
            policy_snapshot_id="snap0", rollout_iteration=1, stable_question_id="q0"
        )
        tid_b = make_tree_instance_id(
            policy_snapshot_id="snap0", rollout_iteration=1, stable_question_id="q0"
        )
        edges = normalize_generated_edges(
            [
                self._edge(tree_instance_id=tid_a),
                self._edge(tree_instance_id=tid_b),
            ],
            snapshot_id="snap0",
            strict=True,
        )
        assert edges[0]["edge_id"] != edges[1]["edge_id"]

    def test_strict_rejects_tree_id_only_record(self):
        # PLAN.md M3: strict normalization requires tree_instance_id
        # specifically; tree_id alone must fail even when present.
        edge = self._edge()
        edge.pop("tree_instance_id")
        edge["tree_id"] = "t0"
        with pytest.raises(ValueError, match="tree_instance_id"):
            normalize_generated_edges([edge], snapshot_id="snap0", strict=True)

    def test_strict_rejects_generic_tree_instance_id(self):
        # PLAN.md M3: a present-but-generic tree_instance_id like "t0" is
        # truthy but structureless — strict mode must reject it.
        with pytest.raises(ValueError, match="canonical tree_instance_id"):
            normalize_generated_edges(
                [self._edge(tree_instance_id="t0")],
                snapshot_id="snap0",
                strict=True,
            )

    @pytest.mark.parametrize(
        "bad_id",
        [
            "t0",
            "snap0",  # snapshot only
            "snap0|iter:1|q:q0",  # missing tiebreaker
            "snap0|q:q0|t:abc",  # missing iter marker
            "snap0|iter:1|t:abc",  # missing question marker
            "wrong|iter:1|q:q0|t:abc",  # snapshot mismatch
            "snap0|iter:|q:q0|t:abc",  # empty iteration
            "snap0|iter:1|q:|t:abc",  # empty question
        ],
    )
    def test_strict_rejects_non_canonical_ids(self, bad_id):
        with pytest.raises(ValueError, match="canonical tree_instance_id"):
            normalize_generated_edges(
                [self._edge(tree_instance_id=bad_id)],
                snapshot_id="snap0",
                strict=True,
            )

    def test_strict_accepts_canonical_builder_id(self):
        tid = make_tree_instance_id(
            policy_snapshot_id="snap0",
            rollout_iteration=1,
            stable_question_id="q0",
        )
        [edge] = normalize_generated_edges(
            [self._edge(tree_instance_id=tid)],
            snapshot_id="snap0",
            strict=True,
        )
        assert edge["edge_id"].startswith("snap0:")

    def test_generic_id_survives_in_non_strict_mode(self):
        # Non-strict compatibility path keeps the legacy fallback for old
        # fixtures — a generic id is not rejected there.
        [edge] = normalize_generated_edges(
            [self._edge(tree_instance_id="t0")],
            snapshot_id="snap0",
            strict=False,
        )
        assert edge["edge_id"].startswith("snap0:")

    def test_mismatching_supplied_edge_id_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            normalize_generated_edges(
                [self._edge(edge_id="snap0:not-the-derived-id")],
                snapshot_id="snap0",
                strict=True,
            )

    def test_matching_supplied_edge_id_is_idempotent(self):
        [first] = normalize_generated_edges(
            [self._edge()], snapshot_id="snap0", strict=True
        )
        # Re-normalizing an already-normalized record must accept its own id.
        [second] = normalize_generated_edges(
            [dict(first)], snapshot_id="snap0", strict=True
        )
        assert second["edge_id"] == first["edge_id"]

    def test_derive_edge_id_matches_historical_formula(self):
        # Pins ID stability: the digest formula is byte-identical to the
        # pre-M3 strict derivation, so no stored edge_id changes value.
        import hashlib

        tid = "snap0|iter:1|q:q0|t:0000-abcd"
        expected_digest = hashlib.blake2b(
            f"{tid}|pg0|c0".encode("utf-8"), digest_size=16
        ).hexdigest()
        assert derive_edge_id(
            snapshot_id="snap0",
            tree_instance_id=tid,
            parent_group_id="pg0",
            child_segment_id="c0",
        ) == f"snap0:{expected_digest}"

    def test_non_strict_keeps_legacy_fallback_chain(self):
        edges = normalize_generated_edges(
            [
                {
                    "question_id": "q0",
                    "gear_segment_id": "seg0",
                    "response_token_ids": [1],
                    "actor_shifted_log_probs": [-0.1],
                }
            ],
            snapshot_id="snap0",
            strict=False,
        )
        assert edges[0]["edge_id"].startswith("snap0:")


class TestManifestCollisionDetection:
    def _edge(self, tree_id: str, child: str, question: str = "q0") -> dict:
        return {
            "tree_id": tree_id,
            "child_segment_id": child,
            "question_id": question,
            "policy_snapshot_id": "snap0",
        }

    def test_distinct_trees_pass(self):
        edges = [
            self._edge("snap0|iter:1|q:q0|t:0", "c0"),
            self._edge("snap0|iter:1|q:q0|t:0", "c1"),
            self._edge("snap0|iter:1|q:q0|t:1", "c0"),
            self._edge("snap0|iter:1|q:q0|t:1", "c1"),
        ]
        ok, details = verify_tree_instance_id_uniqueness(edges)
        assert ok, details

    def test_collision_under_one_tree_id_is_detected(self):
        # Two stochastic trees for the same question merged under one id:
        # the root children repeat their child_segment_id.
        edges = [
            self._edge("snap0:q0-collided", "c0"),
            self._edge("snap0:q0-collided", "c1"),
            self._edge("snap0:q0-collided", "c0"),
            self._edge("snap0:q0-collided", "c1"),
        ]
        ok, details = verify_tree_instance_id_uniqueness(edges)
        assert not ok
        assert any("collided under one id" in d for d in details)

    def test_ambiguous_snapshot_question_identity_is_rejected(self):
        edges = [self._edge("snap0:q0", "c0")]
        ok, details = verify_tree_instance_id_uniqueness(edges)
        assert not ok
        assert any("ambiguous" in d for d in details)

    def _summary_record(self, tree_id: str) -> dict:
        return {
            "tree_summary": {
                "tree_id": tree_id,
                "policy_snapshot_id": "snap0",
                "rollout_iteration": 1,
                "question_id": "q0",
            }
        }

    def test_summary_only_rejects_wrong_snapshot(self):
        ok, details = verify_tree_instance_id_uniqueness(
            [self._summary_record("wrong_snapshot|iter:1|q:q0|t:abc")]
        )
        assert not ok
        assert any("snapshot" in d for d in details)

    def test_summary_only_rejects_wrong_iteration(self):
        ok, details = verify_tree_instance_id_uniqueness(
            [self._summary_record("snap0|iter:999|q:q0|t:abc")]
        )
        assert not ok
        assert any("rollout iteration" in d for d in details)

    def test_summary_only_rejects_wrong_question(self):
        ok, details = verify_tree_instance_id_uniqueness(
            [self._summary_record("snap0|iter:1|q:wrong_question|t:abc")]
        )
        assert not ok
        assert any("question" in d for d in details)

    def test_strict_manifest_update_raises_on_collision(self):
        from recipe.gear_tree.manifest_lifecycle import (
            build_run_manifest,
            update_manifest_from_generated_edges,
        )

        manifest = build_run_manifest(
            tree_policy={"policy_aggregation": "segment_mean"},
            gear_tree_cfg={},
            actor_loss_mode="vdra_segment_mean_ppo",
        )
        collided = [
            {
                "edge_id": f"e{i}",
                "tree_id": "t-collided",
                "parent_group_id": "t-collided#root",
                "child_segment_id": f"c{i % 2}",  # duplicate child ids
                "question_id": "q0",
                "allocated_k": 4,
                "sample_multiplicity": 1,
                "tree_total_segment_count": 4,
                "queue_flush_id": "0",
                "queue_released_segment_count": 4,
                "response_token_ids": [1],
                "actor_shifted_log_probs": [-0.1],
            }
            for i in range(4)
        ]
        with pytest.raises(ValueError, match="P0.H"):
            update_manifest_from_generated_edges(
                manifest, collided, strict=True
            )
        assert manifest.unique_tree_ids_verified is False
