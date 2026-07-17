"""Async (agent-loop) tree rollout for verl >= 0.7 + vLLM >= 0.20.

verl 0.7+ removed the synchronous SPMD ``vLLMRollout``; generation now goes
through an async OpenAI-compatible server driven by ``AsyncLLMServerManager`` and
the ``experimental.agent_loop`` framework. This module re-targets the tree
rollout onto that stack **without touching the algorithm core**:

  * ``AsyncServerSegmentGenerator`` — an async ``segment_fn`` that fires
    ``branch_factor`` concurrent ``server_manager.generate`` calls per node and
    converts each ``TokenOutput`` into a ``SegmentSample``. This is the only new
    engine-coupled code; it is CPU-testable with a mock server manager.
  * ``TreeAgentLoop`` — an ``AgentLoopBase`` (registered ``gear_tree_agent``)
    whose ``run`` builds the whole SPO/GEAR tree via ``async_build_tree``,
    extracts edges (``tree_advantage``), and stashes them in
    ``AgentLoopOutput.extra_fields["gear_tree_edges"]`` so they land in the
    rollout DataProto's ``non_tensor_batch``. The trainer then flattens the
    per-prompt edges into the training batch (``tree_data.edges_to_dataproto``).

Everything downstream — advantages (``compute_tree_update_values``), GEAR gate,
VinePPO MC values, ``treetune_ppo`` loss, reward grading, logging — is unchanged
and shared with the SPMD path.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from recipe.gear_tree.tree_rollout import SegmentSample, async_build_tree
from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.gear_core.reward_function import MathRewardFunction


def _prompt_sticky_key(prompt_ids: Sequence[int]) -> str:
    """Compute a deterministic sticky-routing key from the tail of a prompt.

    P0.1: the previous implementation called ``bytes(list(prompt_ids)[-256:])``
    which raises ``ValueError`` on real vocabulary IDs above 255. blake2b
    accepts arbitrary bytes; we pack each token id as 4 little-endian bytes to
    keep the digest defined for the full 32-bit id space.
    """

    import hashlib

    tail = list(prompt_ids)[-256:]
    buf = b"".join(int(x).to_bytes(4, "little", signed=False) for x in tail)
    return "stick:" + hashlib.blake2b(buf, digest_size=12).hexdigest()


class AsyncServerSegmentGenerator:
    """Async ``segment_fn`` backed by ``AsyncLLMServerManager.generate``."""

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        *,
        base_sampling_params: Optional[Dict[str, Any]] = None,
        free_max_tokens: int = 1024,
    ) -> None:
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.base_sampling_params = dict(base_sampling_params or {})
        self.free_max_tokens = free_max_tokens

    async def _one(
        self,
        prompt_ids: Sequence[int],
        sp: Dict[str, Any],
        *,
        sticky_key: Optional[str] = None,
    ) -> SegmentSample:
        # P0.1: request_id is unique per generation call (concurrent siblings
        # must not collide on the server), sticky_key is the stable branch/
        # session identifier used only for server routing so pilot + its
        # continuation land on the same replica for prefix cache reuse. The
        # legacy bytes(prompt_ids) digest raised on real vocabulary IDs above
        # 255 and gave concurrent siblings the same request id.
        legacy_key = sp.pop("_session_id", None)
        if sticky_key is None:
            sticky_key = legacy_key
        if sticky_key is None:
            sticky_key = _prompt_sticky_key(prompt_ids)
        request_id = f"req:{uuid4().hex}"
        # Forward sticky_key when the wrapped manager supports it; fall back
        # to plain request_id routing on older managers (mock/test doubles).
        try:
            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                sampling_params=sp,
                sticky_key=str(sticky_key),
            )
        except TypeError:
            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                sampling_params=sp,
            )
        tids = list(out.token_ids)
        logps = list(out.log_probs) if getattr(out, "log_probs", None) else None
        cap = sp.get("max_tokens", self.free_max_tokens)
        # Truncated => expandable. Use the server's stop_reason when present,
        # else fall back to "hit the token cap".
        stop_reason = getattr(out, "stop_reason", None)
        truncated = (stop_reason == "length") or (len(tids) >= cap)
        text = self.tokenizer.decode(tids, skip_special_tokens=True)
        return SegmentSample(
            token_ids=tids,
            text=text,
            finish_reason="length" if truncated else "stop",
            logprobs=logps,
        )

    async def segment_fn(
        self,
        prompt_token_ids: Sequence[int],
        branch_factor: int,
        max_tokens: Optional[int],
        *,
        sticky_key: Optional[str] = None,
        sibling_sticky_keys: Optional[Sequence[str]] = None,
    ) -> List[SegmentSample]:
        sp = {
            **self.base_sampling_params,
            "n": 1,
            "logprobs": 1,
            "max_tokens": max_tokens if max_tokens is not None else self.free_max_tokens,
        }
        bf = int(branch_factor)
        if sibling_sticky_keys is not None:
            keys = list(sibling_sticky_keys)
            if len(keys) != bf:
                raise ValueError(
                    f"sibling_sticky_keys length {len(keys)} != branch_factor {bf}"
                )
        else:
            # Concurrent siblings must not collide on a single server: split
            # the parent sticky_key by sibling index. Continuations pass the
            # already-split sibling key explicitly via `sticky_key=` so the
            # pilot's continuation lands on the same replica.
            base = sticky_key if sticky_key is not None else _prompt_sticky_key(prompt_token_ids)
            keys = [f"{base}/s{i}" for i in range(bf)]
        return await asyncio.gather(
            *[
                self._one(prompt_token_ids, dict(sp), sticky_key=keys[i])
                for i in range(bf)
            ]
        )


class SegmentNodeExpander:
    """Adapter exposing ``segment_fn`` as the ``node_expander`` interface used
    by ``gear_core.gear.tv_estimators.ConditionalTVEstimator`` (pilot short
    continuations for the VDRA budget-allocation path)."""

    def __init__(self, segment_generator: AsyncServerSegmentGenerator, tokenizer: Any) -> None:
        self.segment_generator = segment_generator
        self.tokenizer = tokenizer

    async def expand(
        self,
        *,
        current_node: Dict[str, Any],
        prefix: str,
        depth: int,
        max_tokens: int,
        branch_factor: int,
    ) -> List[Dict[str, Any]]:
        prompt_ids = current_node.get("full_token_ids")
        if prompt_ids is None:
            prompt_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        # P0.1: derive sibling sticky keys from a stable node identity so
        # the pilot and its continuation share routing (prefix cache reuse)
        # while concurrent siblings get unique request IDs.
        parent_key = (
            current_node.get("gear_segment_id")
            or current_node.get("vdra_node_id")
            or _prompt_sticky_key(prompt_ids)
        )
        keys = [f"branch:{parent_key}/d{int(depth)}/i{i}" for i in range(int(branch_factor))]
        samples = await self.segment_generator.segment_fn(
            prompt_ids, int(branch_factor), int(max_tokens),
            sibling_sticky_keys=keys,
        )
        nodes: List[Dict[str, Any]] = []
        for i, s in enumerate(samples):
            nodes.append(
                {
                    "text": s.text,
                    "full_text": prefix + s.text,
                    "depth": depth + 1,
                    "finish_reason": s.finish_reason,
                    "sum_logprobs": s.sum_logprobs,
                    "num_tokens": s.num_tokens,
                    "full_token_ids": list(prompt_ids) + list(s.token_ids),
                    "response_token_ids": list(s.token_ids),
                    "actor_shifted_log_probs": list(s.logprobs) if s.logprobs is not None else None,
                    # P0.1: pilot continuations must land on the same server
                    # replica as the original pilot for prefix cache reuse.
                    "vdra_sticky_key": keys[i],
                }
            )
        return nodes


async def build_tree_edges_async(
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    segment_generator: AsyncServerSegmentGenerator,
    reward_fn: MathRewardFunction,
    tree_shape: Sequence[int],
    M: int,
    gear_gate: Any = None,
    tree_update_mode: str = "spo",
    adv_method: str = "rloo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
    only_adv_greater_than_zero: bool = True,
    vineppo_K: int = 0,
    unfinished_penalty: float = 0.0,
    demo_logger: Any = None,
    gear_node_expander: Any = None,
    policy_snapshot_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build one tree via the async server and return its SPO/GEAR edges."""

    def grade_fn(query, response, inst):
        return float(reward_fn(query, response, inst)[0])

    if policy_snapshot_id is not None:
        data_instance = dict(data_instance)
        data_instance["policy_snapshot_id"] = str(policy_snapshot_id)
        data_instance["current_rollout_snapshot_id"] = str(policy_snapshot_id)

    tree = await async_build_tree(
        prompt_text, prompt_token_ids, data_instance,
        tree_shape=tree_shape, M=M,
        segment_fn=segment_generator.segment_fn, grade_fn=grade_fn, gear_gate=gear_gate,
        gear_node_expander=gear_node_expander,
        free_max_tokens=int(getattr(segment_generator, "free_max_tokens", 1024)),
    )

    if vineppo_K > 0:
        from recipe.gear_tree.vineppo_advantage import annotate_tree_with_mc_values

        async def _mc(prefix_ids, K):
            return await segment_generator.segment_fn(prefix_ids, K, None)

        # MC values need async rollouts; run them synchronously within this coroutine.
        await _annotate_mc_async(tree, data_instance, _mc, grade_fn, vineppo_K, unfinished_penalty)

    if demo_logger is not None:
        try:
            demo_logger.log_tree(tree, data_instance.get("_treetune__idx"))
        except Exception:
            pass

    edges = extract_edges_from_tree(
        tree, adv_method=adv_method, only_adv_greater_than_zero=only_adv_greater_than_zero,
        tree_update_mode=tree_update_mode, treepo_global_weight=treepo_global_weight,
        treerl_gamma=treerl_gamma,
    )
    if policy_snapshot_id is not None:
        for edge in edges:
            edge.setdefault("policy_snapshot_id", str(policy_snapshot_id))
    return edges


async def _annotate_mc_async(tree, data_instance, mc_rollout, grade_fn, K, unfinished_penalty):
    """Async version of vineppo_advantage.annotate_tree_with_mc_values."""
    children = tree.get("children") or []
    for child in children:
        await _annotate_mc_async(child, data_instance, mc_rollout, grade_fn, K, unfinished_penalty)
    if children:
        samples = await mc_rollout(tree.get("full_token_ids", []), K)
        rewards = [
            grade_fn(tree.get("full_text", ""), tree.get("full_text", "") + s.text, data_instance)
            if s.finish_reason != "length" else float(unfinished_penalty)
            for s in samples
        ]
        tree["reward"] = sum(rewards) / len(rewards) if rewards else 0.0


def _resolve_external_score_fn_cpu(g: dict):
    """Import ``module:attr`` (default attr ``score_node``) if configured."""
    spec = g.get("external_score_module")
    if not spec:
        return None
    import importlib

    module_name, _, attr = str(spec).partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "score_node")


# P0.2: worker/process-level cache shared by both the CPU and agent-loop
# build paths. Building a scorer per prompt spawns one /models request, HTTP
# client, semaphore, and connection pool per prompt; equal string snapshot
# ids do NOT prove equal weights, so we also stamp a weight_version that the
# trainer can re-verify after every actor update.
_SCORER_CACHE_CPU: Dict[tuple, Any] = {}


def _build_scorer_cpu(g: dict, tokenizer: Any):
    """Build (or reuse) the log-prob scorer without depending on agent-loop imports."""
    api_base = g.get("scorer_api_base")
    if not api_base or tokenizer is None:
        return None
    from recipe.gear_tree.gear_core.gear.vllm_scorer import (
        VLLMLogprobClient,
        fetch_server_weight_version,
        make_lp_scorer,
        resolve_vllm_model_id,
    )

    rollout_snapshot = g.get("policy_snapshot_id")
    scorer_snapshot = g.get("scorer_snapshot_id", rollout_snapshot)
    if rollout_snapshot is not None and scorer_snapshot != rollout_snapshot:
        raise RuntimeError(
            "VDRA scorer snapshot does not match rollout snapshot: "
            f"{scorer_snapshot!r} != {rollout_snapshot!r}"
        )
    api_key = str(g.get("scorer_api_key", "EMPTY"))
    concurrency = int(g.get("scorer_max_concurrency", 64))
    explicit_model = g.get("scorer_model")
    cache_key = (str(api_base), str(explicit_model or ""), api_key, concurrency)
    cached = _SCORER_CACHE_CPU.get(cache_key)
    if cached is None:
        model_id = resolve_vllm_model_id(
            str(api_base),
            explicit_model,
            api_key=api_key,
            timeout=float(g.get("scorer_model_resolve_timeout", 10.0)),
        )
        client = VLLMLogprobClient(
            api_base=str(api_base),
            model=model_id,
            api_key=api_key,
            max_concurrency=concurrency,
        )
        scorer = make_lp_scorer(
            client, lambda text: tokenizer.encode(text, add_special_tokens=False)
        )
        scorer.scorer_model = model_id
        scorer._client = client
        scorer.api_base = str(api_base)
        _SCORER_CACHE_CPU[cache_key] = scorer
    else:
        scorer = cached
    scorer.policy_snapshot_id = rollout_snapshot
    scorer.scorer_snapshot_id = scorer_snapshot
    # P0.3: prefer a server-reported fingerprint over the client snapshot label.
    server_version = None
    try:
        server_version = fetch_server_weight_version(
            str(api_base),
            api_key=api_key,
            timeout=float(g.get("scorer_version_timeout", 5.0)),
        )
    except Exception:
        server_version = None
    scorer.server_weight_version = server_version
    scorer.weight_version_verified = server_version is not None
    scorer.weight_version = server_version or rollout_snapshot
    return scorer


def assert_scorer_matches_rollout(
    scorer,
    rollout_snapshot_id: str,
    *,
    rollout_server_weight_version: Optional[str] = None,
    strict_vdra: bool = True,
) -> None:
    """P0.2 / P0.3 acceptance helper: trainer calls this after every actor update.

    In strict mode with ``rollout_server_weight_version`` supplied, this asserts
    the server-reported scorer fingerprint EQUALS the rollout server's
    fingerprint — matching client-side snapshot strings alone is not proof of
    equal weights (PLAN.md P0.3). If no server-side fingerprint is available
    on the scorer, strict mode fails so the trainer cannot silently proceed
    with unverified weights.
    """
    if scorer is None:
        return
    stamped = getattr(scorer, "weight_version", None)
    server_version = getattr(scorer, "server_weight_version", None)
    verified = bool(getattr(scorer, "weight_version_verified", False))

    if rollout_server_weight_version is not None:
        if not verified or server_version is None:
            if strict_vdra:
                raise RuntimeError(
                    "Scorer weight version is not server-verified: "
                    "strict_vdra requires a server-reported fingerprint "
                    "for both rollout and scorer replicas."
                )
        elif str(server_version) != str(rollout_server_weight_version):
            raise RuntimeError(
                "Scorer server_weight_version does not match rollout server: "
                f"scorer={server_version!r} rollout={rollout_server_weight_version!r}"
            )
        return

    if stamped is None or str(stamped) != str(rollout_snapshot_id):
        raise RuntimeError(
            "Scorer weight_version does not match current rollout snapshot: "
            f"scorer.weight_version={stamped!r} rollout={rollout_snapshot_id!r}"
        )


def _build_gate_cpu(gt: dict, tokenizer: Any = None):
    if not gt.get("gear", {}).get("enabled", False):
        return None
    from recipe.gear_tree.gear_gate import GearGate
    from recipe.gear_tree.calibration import resolve_gear_calibration

    g = resolve_gear_calibration(dict(gt["gear"]))
    if gt.get("policy_snapshot_id") is not None:
        g.setdefault("policy_snapshot_id", gt.get("policy_snapshot_id"))
    scorer = _build_scorer_cpu(g, tokenizer)
    return GearGate(
        epsilon=g.get("epsilon", 0.02), r_max=g.get("r_max", 1.0), gamma=g.get("gamma", 0.9),
        alpha=g.get("alpha", 0.05), k_algorithm=g.get("k_algorithm", "budget_allocation"),
        n_min=g.get("n_min", 1), pilot_branch_factor=g.get("pilot_branch_factor", None), likelihood_samples_per_distribution=g.get("likelihood_samples_per_distribution", 2), root_allocation=g.get("root_allocation", False),
        skip_near_leaf_expand=g.get("skip_near_leaf_expand", True),
        max_depth=len(gt.get("tree_shape", [])) or None, enable_share=g.get("enable_share", False),
        scorer=scorer,
        eps_tail=g.get("eps_tail", 0.0),
        eps_tail_by_depth=g.get("eps_tail_by_depth", None),
        bound_form=g.get("bound_form", "linear"),
        tv_estimator=g.get("tv_estimator", "tanh"),
        tv_first_phase_tokens=g.get("tv_first_phase_tokens", 60),
        tv_second_phase_tokens=g.get("tv_second_phase_tokens", 60),
        queue_count=g.get("queue_count", 4), queue_capacity=g.get("queue_capacity", 8),
        queue_timeout_seconds=g.get("queue_timeout_seconds", 1.0),
        use_residual_budget=g.get("use_residual_budget", True), strict_vdra=g.get("strict_vdra", True), invalid_support_policy=g.get("invalid_support_policy", "error"), budget_mode=g.get("budget_mode", "fixed_main"),
        allocation_proxy=g.get("allocation_proxy", "vdra"),
        allocation_runtime=g.get("allocation_runtime", "online_timeout"),
        artifact_dir=g.get("artifact_dir"),
        eps_tail_calibration_path=g.get("eps_tail_source"),
        eps_tail_calibration_metadata=g.get("eps_tail_calibration_metadata"),
        oracle_rollouts_per_node=g.get("oracle_rollouts_per_node", 16),
        external_score_fn=_resolve_external_score_fn_cpu(g),
        rounding_strategy=g.get("rounding_strategy", "integer_marginal"),
        rounding_seed=g.get("rounding_seed", 0),
        pilot_execution_mode=g.get("pilot_execution_mode", "fresh_iid"),
        weighted_reuse_fallback=g.get("weighted_reuse_fallback", "fresh_iid"),
        representative_weight_mode=g.get("representative_weight_mode", "cluster_multiplicity"),
        terminal_pilot_handling=g.get("terminal_pilot_handling", "include_in_dispersion"),
        rollout_temperature=g.get("rollout_temperature", 1.0),
        rollout_top_p=g.get("rollout_top_p", 1.0),
    )


# --------------------------------------------------------------------------- #
# TreeAgentLoop — registered agent for verl's agent-loop framework (GPU path).
# --------------------------------------------------------------------------- #
try:  # keep CPU-importable when agent_loop isn't installed
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

    @register("gear_tree_agent")  # pragma: no cover - requires async server + GPU
    class TreeAgentLoop(AgentLoopBase):
        """Build a full SPO/GEAR tree per prompt; emit edges via extra_fields."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            gt = dict(self.config.get("gear_tree", {}))
            gear_cfg = dict(gt.get("gear", {}))
            # P0.3: overwrite, don't setdefault. Nested VDRA config often carries
            # stale rollout_temperature/rollout_top_p from an earlier ablation;
            # the strict gate must validate the *actual* sampling distribution
            # used by the async rollout server, which is rollout_config.*.
            actual_temp = float(self.rollout_config.temperature)
            actual_top_p = float(self.rollout_config.top_p)
            gear_cfg["rollout_temperature"] = actual_temp
            gear_cfg["rollout_top_p"] = actual_top_p
            # Enforce the tanh-TV estimator's distributional prerequisites.
            # The scorer implements the untransformed p(a|s); if the rollout
            # server samples under (temperature, top_p) != (1, 1) then scorer
            # and rollout are using different distributions and the TV
            # estimate becomes invalid. Until the scorer explicitly implements
            # the matching transformed distribution, refuse to start.
            if str(gear_cfg.get("tv_estimator", "tanh")) == "tanh" and bool(
                gear_cfg.get("enabled", False)
            ):
                if actual_temp != 1.0 or actual_top_p != 1.0:
                    raise RuntimeError(
                        "VDRA tanh-TV estimator requires rollout temperature=1.0 "
                        "and top_p=1.0 (see PLAN.md P0.3); got "
                        f"temperature={actual_temp}, top_p={actual_top_p}."
                    )
            gt["gear"] = gear_cfg
            self._gt = gt
            self._gen = AsyncServerSegmentGenerator(
                self.server_manager, self.tokenizer,
                base_sampling_params={
                    "temperature": self.rollout_config.temperature,
                    "top_p": self.rollout_config.top_p,
                },
                free_max_tokens=self.rollout_config.response_length,
            )
            self._reward_fn = MathRewardFunction(
                answer_prefix=gt.get("answer_prefix", "# Answer\n"),
                use_minerva_few_shot_prompt=gt.get("use_minerva_few_shot_prompt", False),
            )
            self._gate = _build_gate(gt, tokenizer=self.tokenizer)
            self._node_expander = SegmentNodeExpander(self._gen, self.tokenizer)

        async def run(self, sampling_params: dict, **kwargs) -> "AgentLoopOutput":
            messages = list(kwargs["raw_prompt"])
            prompt_ids = await self.apply_chat_template(messages)
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

            rm = kwargs.get("reward_model", {}) or {}
            extra = kwargs.get("extra_info", {}) or {}
            # P0.1: prefer per-sample non_tensor_batch kwargs; fall back to the
            # config only if it carries a real snapshot id. Never accept the
            # "rollout_step:unknown" sentinel here — it means the trainer did
            # not propagate the snapshot and would corrupt edge provenance.
            snapshot_id = (
                kwargs.get("policy_snapshot_id")
                or kwargs.get("current_rollout_snapshot_id")
                or self._gt.get("policy_snapshot_id")
            )
            if not snapshot_id or snapshot_id == "rollout_step:unknown":
                raise RuntimeError(
                    "TreeAgentLoop.run received no policy_snapshot_id via "
                    "non_tensor_batch kwargs or gear_tree_config; trainer must "
                    "populate gen_batch.non_tensor_batch['policy_snapshot_id'] "
                    "before generate_sequences (see PLAN.md P0.1)."
                )
            snapshot_id = str(snapshot_id)
            data_instance = {
                "problem": extra.get("problem") or rm.get("problem"),
                "answer": rm.get("ground_truth"),
                "reward_model": rm,
                "_treetune__idx": kwargs.get("index"),
                "policy_snapshot_id": snapshot_id,
                "current_rollout_snapshot_id": snapshot_id,
            }

            # P0.2 / P0.3: rebind the gate/scorer to THIS rollout's snapshot
            # before touching the tree builder. The gate was constructed in
            # __init__ from static config; without a per-rollout re-bind the
            # scorer keeps its init-time weight_version even after the actor
            # has stepped, and strict-mode invariants would compare stale
            # state.
            if self._gate is not None:
                server_weight_version = (
                    kwargs.get("rollout_server_weight_version")
                    or kwargs.get("scorer_server_weight_version")
                )
                self._gate.bind_snapshot(
                    snapshot_id,
                    weight_version=server_weight_version,
                    weight_version_verified=server_weight_version is not None,
                )
                # P0.4: wire the terminal grader so terminal pilots contribute
                # observed reward differences to the dispersion estimate.
                reward_fn = self._reward_fn

                def _terminal_grader(node):
                    text = node.get("full_text") or node.get("text") or ""
                    return float(reward_fn(prompt_text, text, data_instance)[0])

                self._gate.set_terminal_reward_fn(_terminal_grader)

            edges = await build_tree_edges_async(
                prompt_text, prompt_ids, data_instance,
                segment_generator=self._gen, reward_fn=self._reward_fn,
                tree_shape=self._gt.get("tree_shape", [6, 6, 6]),
                M=self._gt.get("segment_length", 100), gear_gate=self._gate,
                tree_update_mode=self._gt.get("tree_update_mode", "spo"),
                adv_method=self._gt.get("adv_method", "rloo"),
                treepo_global_weight=self._gt.get("treepo_global_weight", 0.5),
                treerl_gamma=self._gt.get("treerl_gamma", 0.9),
                only_adv_greater_than_zero=self._gt.get("only_adv_greater_than_zero", True),
                vineppo_K=self._gt.get("vineppo_K", 0),
                gear_node_expander=self._node_expander,
                policy_snapshot_id=snapshot_id,
            )

            # Placeholder response (first edge) keeps the AgentLoopOutput schema
            # valid; the real training rows are the edges in extra_fields.
            resp = (edges[0].get("response_token_ids") if edges else [self.tokenizer.eos_token_id]) or [
                self.tokenizer.eos_token_id
            ]
            resp = resp[: self.rollout_config.response_length]
            # P1.2: skip placeholder reward computation. The custom trainer
            # scores tree edges, not this placeholder — grading it wastes
            # reward-model calls (and, for MathReward, can flag legitimate
            # tree edges as wrong).
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=resp,
                response_mask=[1] * len(resp),
                num_turns=2,
                reward_score=0.0,
                metrics=AgentLoopMetrics(),
                extra_fields={"gear_tree_edges": edges},
            )

    def _build_gate(gt: dict, tokenizer: Any = None):
        if not gt.get("gear", {}).get("enabled", False):
            return None
        from recipe.gear_tree.gear_gate import GearGate

        from recipe.gear_tree.calibration import resolve_gear_calibration

        g = resolve_gear_calibration(dict(gt["gear"]))
        if gt.get("policy_snapshot_id") is not None:
            g.setdefault("policy_snapshot_id", gt.get("policy_snapshot_id"))
        scorer = _build_scorer(g, tokenizer)
        # Defaults here must match the GearGate signature so a missing config
        # key behaves identically no matter which entry point built the gate.
        return GearGate(
            epsilon=g.get("epsilon", 0.02), r_max=g.get("r_max", 1.0), gamma=g.get("gamma", 0.9),
            alpha=g.get("alpha", 0.05), k_algorithm=g.get("k_algorithm", "budget_allocation"),
            n_min=g.get("n_min", 1), pilot_branch_factor=g.get("pilot_branch_factor", None), likelihood_samples_per_distribution=g.get("likelihood_samples_per_distribution", 2), root_allocation=g.get("root_allocation", False),
            skip_near_leaf_expand=g.get("skip_near_leaf_expand", True),
            max_depth=len(gt.get("tree_shape", [])) or None, enable_share=g.get("enable_share", False),
            scorer=scorer,
            eps_tail=g.get("eps_tail", 0.0),
            eps_tail_by_depth=g.get("eps_tail_by_depth", None),
            bound_form=g.get("bound_form", "linear"),
            tv_estimator=g.get("tv_estimator", "tanh"),
            tv_first_phase_tokens=g.get("tv_first_phase_tokens", 60),
            tv_second_phase_tokens=g.get("tv_second_phase_tokens", 60),
            queue_count=g.get("queue_count", 4), queue_capacity=g.get("queue_capacity", 8),
            queue_timeout_seconds=g.get("queue_timeout_seconds", 1.0),
            use_residual_budget=g.get("use_residual_budget", True), strict_vdra=g.get("strict_vdra", True), invalid_support_policy=g.get("invalid_support_policy", "error"), budget_mode=g.get("budget_mode", "fixed_main"),
            allocation_proxy=g.get("allocation_proxy", "vdra"),
            allocation_runtime=g.get("allocation_runtime", "online_timeout"),
            artifact_dir=g.get("artifact_dir"),
            eps_tail_calibration_path=g.get("eps_tail_source"),
            eps_tail_calibration_metadata=g.get("eps_tail_calibration_metadata"),
            oracle_rollouts_per_node=g.get("oracle_rollouts_per_node", 16),
            external_score_fn=_resolve_external_score_fn(g),
            rounding_strategy=g.get("rounding_strategy", "integer_marginal"),
            rounding_seed=g.get("rounding_seed", 0),
            pilot_execution_mode=g.get("pilot_execution_mode", "fresh_iid"),
            weighted_reuse_fallback=g.get("weighted_reuse_fallback", "fresh_iid"),
            representative_weight_mode=g.get("representative_weight_mode", "cluster_multiplicity"),
            terminal_pilot_handling=g.get("terminal_pilot_handling", "include_in_dispersion"),
        )

    def _resolve_external_score_fn(g: dict):
        """Import ``module:attr`` (default attr ``score_node``) if configured."""
        spec = g.get("external_score_module")
        if not spec:
            return None
        import importlib

        module_name, _, attr = str(spec).partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr or "score_node")

    # P0.2: worker-level cache. Building a scorer per prompt spawns one /models
    # request, HTTP client, semaphore, and connection pool per prompt; that
    # both wastes resources and makes it impossible to verify that scorer and
    # rollout share the same loaded weights. Key by (endpoint, model, api_key,
    # concurrency) so different rollouts in the same process still share.
    _SCORER_CACHE: Dict[tuple, Any] = {}

    def _build_scorer(g: dict, tokenizer: Any):
        """Build (or reuse) the log-prob scorer for share / budget-allocation paths.

        ``scorer_api_base`` points at an OpenAI-compatible vLLM server (the
        agent-loop stack already runs one); without it the gate runs with
        """
        api_base = g.get("scorer_api_base")
        if not api_base or tokenizer is None:
            return None
        from recipe.gear_tree.gear_core.gear.vllm_scorer import (
            VLLMLogprobClient,
            fetch_server_weight_version,
            make_lp_scorer,
            resolve_vllm_model_id,
        )

        rollout_snapshot = g.get("policy_snapshot_id")
        scorer_snapshot = g.get("scorer_snapshot_id", rollout_snapshot)
        # Per PLAN.md P0.2 the snapshot IDs must match; matching *strings* is a
        # necessary but not sufficient signal, so we also stamp a
        # weight_version below that the trainer can re-verify after each
        # actor update.
        if rollout_snapshot is not None and scorer_snapshot != rollout_snapshot:
            raise RuntimeError(
                "VDRA scorer snapshot does not match rollout snapshot: "
                f"{scorer_snapshot!r} != {rollout_snapshot!r}"
            )
        api_key = str(g.get("scorer_api_key", "EMPTY"))
        concurrency = int(g.get("scorer_max_concurrency", 64))
        explicit_model = g.get("scorer_model")

        cache_key = (str(api_base), str(explicit_model or ""), api_key, concurrency)
        cached = _SCORER_CACHE.get(cache_key)
        if cached is None:
            model_id = resolve_vllm_model_id(
                str(api_base),
                explicit_model,
                api_key=api_key,
                timeout=float(g.get("scorer_model_resolve_timeout", 10.0)),
            )
            client = VLLMLogprobClient(
                api_base=str(api_base),
                model=model_id,
                api_key=api_key,
                max_concurrency=concurrency,
            )
            scorer = make_lp_scorer(
                client, lambda text: tokenizer.encode(text, add_special_tokens=False)
            )
            scorer.scorer_model = model_id
            scorer._client = client  # retained so shutdown can aclose() it
            scorer.api_base = str(api_base)
            _SCORER_CACHE[cache_key] = scorer
        else:
            scorer = cached
        # Refresh the per-run snapshot stamps every time — the trainer bumps
        # policy_snapshot_id after each actor update, and the cached scorer
        # must reflect the current expected weight version.
        scorer.policy_snapshot_id = rollout_snapshot
        scorer.scorer_snapshot_id = scorer_snapshot
        # P0.3: attempt a real server-side handshake before falling back to
        # the client-side snapshot label. The server may respond with a real
        # fingerprint (checkpoint revision, actor update number); the trainer
        # then compares this against the rollout server's fingerprint via
        # assert_scorer_matches_rollout(). If no fingerprint is available we
        # keep the snapshot string but flag it as unverified so downstream
        # code (run manifest) does NOT report version_verified=True.
        server_version = None
        try:
            server_version = fetch_server_weight_version(
                str(api_base),
                api_key=api_key,
                timeout=float(g.get("scorer_version_timeout", 5.0)),
            )
        except Exception:
            server_version = None
        scorer.server_weight_version = server_version
        scorer.weight_version_verified = server_version is not None
        scorer.weight_version = server_version or rollout_snapshot
        return scorer

    async def _close_cached_scorers() -> None:
        """Close every cached scorer's HTTP client. Call at worker shutdown."""
        while _SCORER_CACHE:
            _, scorer = _SCORER_CACHE.popitem()
            client = getattr(scorer, "_client", None)
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # pragma: no cover
                    pass

except Exception:  # pragma: no cover
    TreeAgentLoop = None  # type: ignore

    def _build_gate(gt, tokenizer=None):  # type: ignore
        return _build_gate_cpu(gt, tokenizer=tokenizer)


def collect_tree_edges(dataproto) -> List[Dict[str, Any]]:
    """Flatten per-prompt ``gear_tree_edges`` (non_tensor_batch) into one list."""
    edges: List[Dict[str, Any]] = []
    per_prompt = dataproto.non_tensor_batch.get("gear_tree_edges")
    if per_prompt is None:
        raise KeyError("rollout output has no 'gear_tree_edges' (is agent.name=gear_tree_agent?)")
    for item in per_prompt:
        if item:
            edges.extend(item)
    return edges
