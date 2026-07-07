#!/usr/bin/env python3
"""Analysis-only pruning diagnostics for tree-policy runs.

This entry point never updates model weights and never calls the training
runtime.  Use `--backend replay` for local/offline inspection of saved trees.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from treetune.gear.pruning_controller import (  # noqa: E402
    PruningTraceRecord,
    format_trace_record,
    records_from_analysis_node,
    summarize_records,
    trace_records_from_matrices,
)
from treetune.gear.thresholds import ThresholdConfig  # noqa: E402
from treetune.gear.tree_policy_logging import (  # noqa: E402
    branch_factors_from_shape,
    build_run_manifest,
    iter_tree_nodes,
    print_run_banner,
    render_full_tree_markdown,
    serialize_full_tree,
    write_json,
)


def _load_json_or_jsonl(path: Path) -> Dict[str, Any]:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            return obj.get("tree") or obj.get("full_tree") or obj
        raise ValueError(f"{path} has no JSONL records")
    obj = json.loads(text)
    return obj.get("tree") or obj.get("full_tree") or obj


def _default_prompt() -> str:
    return "Solve the problem step by step: What is 2 + 3?"


def _node_id(parent_id: str, depth: int, child_idx: int) -> str:
    return f"{parent_id}/{depth}-{child_idx}"


def _candidate_suffix(builder: str, depth: int, child_idx: int) -> str:
    if builder == "treepo_fixed_step":
        return f"\nStep chunk {depth}.{child_idx}: continue with a fixed-size reasoning segment."
    if builder == "treerl_entropy":
        return f"\nEntropy branch {depth}.{child_idx}: expand from a high-uncertainty token."
    return f"\nSPO step {depth}.{child_idx}: derive the next reasoning step."


def build_synthetic_tree(
    *,
    builder: str,
    prompt: str,
    tree_shape: str,
    tree_m: int,
) -> Dict[str, Any]:
    branch_factors = branch_factors_from_shape(tree_shape)
    root: Dict[str, Any] = {
        "text": prompt,
        "full_text": prompt,
        "depth": 0,
        "segment_id": "0",
        "segmentation_type": builder,
        "segment_token_budget": int(tree_m),
        "children": [],
    }

    def expand(parent: Dict[str, Any], depth: int) -> None:
        if depth >= len(branch_factors):
            return
        branch_factor = branch_factors[depth]
        children = []
        parent_id = str(parent.get("segment_id", "0"))
        for idx in range(branch_factor):
            text = _candidate_suffix(builder, depth, idx)
            child = {
                "text": text,
                "full_text": parent["full_text"] + text,
                "depth": depth + 1,
                "segment_id": _node_id(parent_id, depth, idx),
                "segmentation_type": builder,
                "segment_token_budget": int(tree_m),
                "children": [],
            }
            children.append(child)
        parent["children"] = children
        for child in children:
            expand(child, depth + 1)

    expand(root, 0)
    return root


def _hash_prob(seed: str, support_idx: int) -> float:
    value = sum(ord(ch) for ch in f"{seed}:{support_idx}")
    return 0.1 + float((value % 997) + 1) / 997.0


def add_deterministic_probability_evidence(
    tree: Dict[str, Any],
    *,
    threshold_cfg: ThresholdConfig,
) -> None:
    for node in iter_tree_nodes(tree):
        children = list(node.get("children") or [])
        if len(children) < 2:
            continue
        support_size = min(len(children), 4)
        prob_matrix = []
        for child in children:
            raw = [
                _hash_prob(str(child.get("full_text", "")), support_idx)
                for support_idx in range(support_size)
            ]
            total = sum(raw)
            prob_matrix.append([value / total for value in raw])
        pair_tvs = {}
        value_gaps = {}
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                tv = 0.5 * sum(
                    abs(prob_matrix[i][k] - prob_matrix[j][k])
                    for k in range(support_size)
                )
                pair_tvs[f"{i},{j}"] = tv
                value_gaps[f"{i},{j}"] = abs(
                    float(children[i].get("reward", 0.0) or 0.0)
                    - float(children[j].get("reward", 0.0) or 0.0)
                )
        predicted_k = max(
            1,
            len(children)
            - sum(1 for tv in pair_tvs.values() if tv < threshold_cfg.epsilon),
        )
        node["prunning_analysis"] = {
            "default_branch_factor": len(children),
            "predicted_k": predicted_k,
            "prob_matrix": prob_matrix,
            "pair_tvs": pair_tvs,
            "value_gaps": value_gaps,
            "duplicate_tv_threshold": threshold_cfg.epsilon,
        }


def _records_from_tree(
    tree: Mapping[str, Any],
    *,
    threshold_cfg: ThresholdConfig,
    limit_nodes: int,
) -> List[PruningTraceRecord]:
    records: List[PruningTraceRecord] = []
    inspected = 0
    for node in iter_tree_nodes(tree):
        if len(node.get("children") or []) < 2 and not (
            node.get("prunning_analysis") or node.get("pruning_analysis")
        ):
            continue
        records.extend(records_from_analysis_node(node, threshold_cfg=threshold_cfg))
        inspected += 1
        if limit_nodes > 0 and inspected >= limit_nodes:
            break
    return records


def _mark_after_k_algorithm(tree: Dict[str, Any], records: Sequence[PruningTraceRecord]) -> Dict[str, Any]:
    tree = json.loads(json.dumps(tree, default=str))
    by_node: Dict[str, List[PruningTraceRecord]] = {}
    for record in records:
        by_node.setdefault(record.node_id, []).append(record)
    for node in iter_tree_nodes(tree):
        node_id = str(node.get("gear_segment_id") or node.get("segment_id") or "root")
        node_records = by_node.get(node_id)
        if not node_records:
            continue
        node["prunning_trace_count"] = len(node_records)
        node["prunning_duplicate_pairs"] = sum(1 for rec in node_records if rec.duplicate)
        node["prunning_prune_candidates"] = sum(
            1 for rec in node_records if rec.prune_candidate
        )
        predicted = [rec.predicted_k for rec in node_records if rec.predicted_k is not None]
        if predicted:
            node["gear_predicted_k"] = min(predicted)
    return tree


async def _build_vllm_tree(args, threshold_cfg: ThresholdConfig) -> Dict[str, Any]:
    from treetune.gear.vllm_scorer import VLLMLogprobClient

    api_base = args.api_base or os.environ.get("APP_OPENAI_VLLM_API_BASE")
    if not api_base:
        raise RuntimeError(
            "--backend vllm requires --api-base or APP_OPENAI_VLLM_API_BASE"
        )
    client = VLLMLogprobClient(
        api_base=api_base,
        model=args.model,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency,
    )
    tree = build_synthetic_tree(
        builder=args.builder,
        prompt=args.prompt or _default_prompt(),
        tree_shape=args.tree_shape,
        tree_m=args.tree_m,
    )
    try:
        for node in iter_tree_nodes(tree):
            children = list(node.get("children") or [])
            if len(children) < 2:
                continue
            support = [child.get("text", "") for child in children]
            prob_matrix = []
            for child in children:
                row = []
                for suffix in support:
                    logps = await client.prompt_logprobs(
                        child.get("full_text", "") + suffix
                    )
                    finite = [float(lp) for lp in logps if lp is not None and math.isfinite(float(lp))]
                    row.append(math.exp(sum(finite[-max(1, len(suffix.split())) :])))
                total = sum(row) or 1.0
                prob_matrix.append([value / total for value in row])
            pair_tvs = {}
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    pair_tvs[f"{i},{j}"] = 0.5 * sum(
                        abs(prob_matrix[i][k] - prob_matrix[j][k])
                        for k in range(len(support))
                    )
            node["prunning_analysis"] = {
                "default_branch_factor": len(children),
                "predicted_k": max(
                    1,
                    len(children)
                    - sum(1 for tv in pair_tvs.values() if tv < threshold_cfg.epsilon),
                ),
                "prob_matrix": prob_matrix,
                "pair_tvs": pair_tvs,
                "duplicate_tv_threshold": threshold_cfg.epsilon,
            }
    finally:
        await client.aclose()
    return tree


def _build_transformers_tree(args, threshold_cfg: ThresholdConfig) -> Dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "--backend transformers requires transformers and torch. "
            "Use --backend replay for dependency-free local analysis."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    tree = build_synthetic_tree(
        builder=args.builder,
        prompt=args.prompt or _default_prompt(),
        tree_shape=args.tree_shape,
        tree_m=args.tree_m,
    )

    def score(prompt: str, suffix: str) -> float:
        encoded = tokenizer(prompt + suffix, return_tensors="pt")
        prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        with torch.no_grad():
            logits = model(**encoded).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = encoded["input_ids"][:, 1:]
        token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
        tail = token_logps[max(prompt_len - 1, 0) :]
        return float(tail.sum().item())

    for node in iter_tree_nodes(tree):
        children = list(node.get("children") or [])
        if len(children) < 2:
            continue
        support = [child.get("text", "") for child in children]
        prob_matrix = []
        for child in children:
            raw = [math.exp(score(child.get("full_text", ""), suffix)) for suffix in support]
            total = sum(raw) or 1.0
            prob_matrix.append([value / total for value in raw])
        pair_tvs = {}
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pair_tvs[f"{i},{j}"] = 0.5 * sum(
                    abs(prob_matrix[i][k] - prob_matrix[j][k])
                    for k in range(len(support))
                )
        node["prunning_analysis"] = {
            "default_branch_factor": len(children),
            "predicted_k": max(
                1,
                len(children)
                - sum(1 for tv in pair_tvs.values() if tv < threshold_cfg.epsilon),
            ),
            "prob_matrix": prob_matrix,
            "pair_tvs": pair_tvs,
            "duplicate_tv_threshold": threshold_cfg.epsilon,
        }
    return tree


def _write_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Sequence[PruningTraceRecord],
) -> None:
    lines = [
        "# Prunning Analysis Report\n\n",
        "## Run Manifest\n\n",
        "```json\n",
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        "\n```\n\n",
        "## Summary\n\n",
        "```json\n",
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        "\n```\n\n",
        "## Trace Preview\n\n",
    ]
    for record in records[:20]:
        lines.append("```text\n")
        lines.append(format_trace_record(record))
        lines.append("\n```\n\n")
    path.write_text("".join(lines))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("replay", "transformers", "vllm"), default="replay")
    parser.add_argument("--builder", choices=("spo_step", "treepo_fixed_step", "treerl_entropy"), default="spo_step")
    parser.add_argument("--input-tree", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tree-shape", default="666")
    parser.add_argument("--tree-m", type=int, default=600)
    parser.add_argument("--k-algorithm", default="hierarchical")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--limit-nodes", type=int, default=8)
    parser.add_argument("--synthetic-replay", action="store_true", help="build a deterministic local tree for replay mode")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    threshold_cfg = ThresholdConfig(epsilon=args.epsilon, gamma=args.gamma)
    run_name = args.run_name or f"{args.backend}-{args.builder}-{int(time.time())}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        algorithm_name=f"prunning_{args.builder}",
        segmentation_type=args.builder,
        allocation_type="none",
        pruning_enabled=True,
        allocation_enabled=False,
        k_algorithm=args.k_algorithm,
        tree_shape=args.tree_shape,
        tree_m=args.tree_m,
        backend=args.backend,
        training=False,
    )
    print_run_banner(manifest, prefix="[prunning-analysis]")
    print(f"[prunning-analysis] builder={args.builder}", flush=True)
    print(f"[prunning-analysis] k_algorithm={args.k_algorithm}", flush=True)
    print("[prunning-analysis] allocation=false pruning=true", flush=True)
    print(
        f"[prunning-analysis] tree_shape={args.tree_shape} tree_m={args.tree_m}",
        flush=True,
    )

    if args.backend == "replay":
        if args.input_tree is not None:
            tree = _load_json_or_jsonl(args.input_tree)
        elif args.synthetic_replay:
            tree = build_synthetic_tree(
                builder=args.builder,
                prompt=args.prompt or _default_prompt(),
                tree_shape=args.tree_shape,
                tree_m=args.tree_m,
            )
            add_deterministic_probability_evidence(tree, threshold_cfg=threshold_cfg)
        else:
            raise RuntimeError(
                "--backend replay requires --input-tree or --synthetic-replay"
            )
    elif args.backend == "transformers":
        tree = _build_transformers_tree(args, threshold_cfg)
    else:
        tree = asyncio.run(_build_vllm_tree(args, threshold_cfg))

    full_tree_before = serialize_full_tree(tree)
    records = _records_from_tree(
        full_tree_before, threshold_cfg=threshold_cfg, limit_nodes=args.limit_nodes
    )
    for record in records:
        print(format_trace_record(record), flush=True)

    full_tree_after = _mark_after_k_algorithm(full_tree_before, records)
    summary = summarize_records(records)

    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "prunning_summary.json", summary)
    write_json(out_dir / "full_tree_before.json", full_tree_before)
    write_json(out_dir / "full_tree_after_k_algorithm.json", full_tree_after)
    (out_dir / "full_tree_before.md").write_text(
        render_full_tree_markdown(full_tree_before, tree_idx=1, question_id="analysis")
    )
    with (out_dir / "prunning_trace.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")
    _write_report(out_dir / "report.md", manifest=manifest, summary=summary, records=records)
    print(f"[prunning-analysis] wrote outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
