from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from treetune.gear.tree_policy_logging import (
    render_full_tree_markdown,
    serialize_full_tree,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "prunning_analysis" / "run_prunning_analysis.py"


def test_prunning_analysis_script_is_analysis_only():
    text = SCRIPT.read_text()
    forbidden = [
        "run_iteration_loop",
        "Deepspeed",
        "DeepSpeed",
        "treetune.trainers",
        "BaseTrainer",
        "PolicyTrainer",
        "update_policy",
        "save_checkpoint",
    ]
    for needle in forbidden:
        assert needle not in text


def test_replay_backend_writes_required_artifacts(tmp_path):
    tree_path = tmp_path / "tree.json"
    long_text = "root " + ("x" * 300)
    tree = {
        "text": long_text,
        "full_text": long_text,
        "depth": 0,
        "segment_id": "0",
        "children": [
            {"text": " child a " + ("a" * 260), "full_text": long_text + " child a " + ("a" * 260), "depth": 1, "segment_id": "0/0", "children": []},
            {"text": " child b " + ("b" * 260), "full_text": long_text + " child b " + ("b" * 260), "depth": 1, "segment_id": "0/1", "children": []},
        ],
        "prunning_analysis": {
            "default_branch_factor": 2,
            "predicted_k": 1,
            "prob_matrix": [[0.7, 0.3], [0.45, 0.55]],
            "pair_tvs": {"0,1": 0.25},
            "value_gaps": {"0,1": 0.2},
            "reward_variance": 0.0625,
            "duplicate_tv_threshold": 0.02,
        },
    }
    tree_path.write_text(json.dumps(tree))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "replay", "--input-tree", str(tree_path), "--output-dir", str(tmp_path), "--run-name", "case"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[prunning-analysis] mode=analysis_only training=false" in result.stdout
    assert "[prunning-analysis] backend=replay" in result.stdout
    assert "[prunning-analysis] builder=spo_step" in result.stdout
    out_dir = tmp_path / "case"
    for filename in ["run_manifest.json", "prunning_trace.jsonl", "prunning_summary.json", "full_tree_before.json", "full_tree_after_k_algorithm.json", "report.md"]:
        assert (out_dir / filename).exists()
    trace = json.loads((out_dir / "prunning_trace.jsonl").read_text().splitlines()[0])
    for key in ["p_x", "p_y", "value_gap", "tv", "value_upper_bound", "reward_variance", "predicted_k", "prune_candidate"]:
        assert key in trace
    assert trace["p_x"] == [0.7, 0.3]
    assert trace["p_y"] == [0.45, 0.55]
    before = json.loads((out_dir / "full_tree_before.json").read_text())
    assert before["text"] == long_text
    assert len(before["children"][0]["text"]) > 250


def test_replay_backend_reports_unavailable_probability_fields(tmp_path):
    tree_path = tmp_path / "tree_missing_prob.json"
    tree_path.write_text(json.dumps({"text": "root", "full_text": "root", "depth": 0, "segment_id": "0", "children": [{"text": "a"}, {"text": "b"}]}))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "replay", "--input-tree", str(tree_path), "--output-dir", str(tmp_path), "--run-name", "missing"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    trace = json.loads((tmp_path / "missing" / "prunning_trace.jsonl").read_text())
    assert trace["unavailable_fields"] == ["prob_matrix", "pair_tvs"]
    assert "unavailable_fields=prob_matrix,pair_tvs" in result.stdout


def test_full_tree_serializer_keeps_full_text_without_truncation():
    long_text = "prefix-" + ("z" * 500)
    tree = {"text": long_text, "full_text": long_text, "_request_object": {"id": "q1"}, "children": [{"text": "child-" + ("c" * 400), "full_text": long_text + " child-" + ("c" * 400)}]}
    serialized = serialize_full_tree(tree)
    rendered = render_full_tree_markdown(serialized, tree_idx=1, question_id="q1")
    assert serialized["text"] == long_text
    assert serialized["children"][0]["text"] == "child-" + ("c" * 400)
    assert long_text in rendered
    assert "..." not in rendered
