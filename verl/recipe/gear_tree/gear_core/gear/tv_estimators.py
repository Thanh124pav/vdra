"""TV estimators for simulation-lemma budget allocation.

The estimators build and cache a conditional log-probability matrix
``log P(ss_k2 | ss_i1)``.  Pairwise TV values are computed from this matrix,
so each unique (prefix, continuation) scoring request is made at most once per
estimator instance.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Node = Dict[str, Any]
from treetune.gear.budget_allocation import reward_variance_from_pair_tvs


PairKey = Tuple[int, int]


def _pairwise_tv_tanh_with_count(
    logps_i: Sequence[float], logps_j: Sequence[float]
) -> Tuple[float, int]:
    """Likelihood-based short-horizon TV estimator (Summary.md §9).

    Uses the identity |a-b|/(a+b) = |tanh((log a - log b)/2)| so the estimate
    depends only on the log-probability *ratio* of each sampled block under the
    two conditional distributions — full-sequence probabilities like exp(-60)
    never appear, which keeps the estimator numerically meaningful.

        D_hat = mean_z |tanh((log P_i(z) - log P_j(z)) / 2)|,

    averaged over blocks z drawn from the mixture (Z_i ∪ Z_j when per-pair
    supports are available).  A block with zero probability under exactly one
    distribution contributes 1 (disjoint support).
    """

    vals: List[float] = []
    for a, b in zip(logps_i, logps_j):
        try:
            a = float(a)
            b = float(b)
        except (TypeError, ValueError):
            continue
        finite_a = math.isfinite(a)
        finite_b = math.isfinite(b)
        if not finite_a or not finite_b:
            if finite_a != finite_b:
                vals.append(1.0)
            continue
        vals.append(abs(math.tanh((a - b) / 2.0)))
    if not vals:
        return 0.0, 0
    return sum(vals) / len(vals), len(vals)


def pairwise_tv_tanh(logps_i: Sequence[float], logps_j: Sequence[float]) -> float:
    value, _ = _pairwise_tv_tanh_with_count(logps_i, logps_j)
    return value


@dataclass
class TVSample:
    first: Node
    second: Node


@dataclass
class TVEstimateResult:
    samples: List[TVSample]
    logp_matrix: List[List[float]]
    prob_matrix: List[List[float]]
    pair_tvs: Dict[PairKey, float]
    dispersion_C: float
    dispersion_C_continuation: float = 0.0
    dispersion_C_terminal: float = 0.0
    dispersion_C_cross: float = 0.0

    @property
    def reward_variance(self) -> float:
        return self.dispersion_C
    candidates: List[Node] = field(default_factory=list)
    predicted_k: int = 0
    unique_candidates: List[Node] = field(default_factory=list)
    duplicate_pairs: List[PairKey] = field(default_factory=list)
    # Pilots that terminated (EOS) inside the first phase. They are complete
    # trajectories: the caller must attach them as graded leaf children counted
    # against the node's branch budget instead of scoring TV on them.
    shortcut_candidates: List[Node] = field(default_factory=list)
    # Second-phase support blocks. Exposed so callers can charge their
    # generation cost (tokens/requests) to the pilot overhead accounting.
    support_nodes: List[Node] = field(default_factory=list)
    cluster_id_per_pilot: List[int] = field(default_factory=list)
    representative_index_per_cluster: Dict[int, int] = field(default_factory=dict)
    cluster_size: Dict[int, int] = field(default_factory=dict)


class ConditionalTVEstimator:
    """Generate TV samples and compute pairwise TV from a cached matrix."""

    def __init__(
        self,
        *,
        scorer: Any,
        node_expander: Any,
        gamma: float,
        mode: str = "subnode",
        n_tv_estimates: Optional[int] = None,
        first_phase_tokens: int = 120,
        second_phase_tokens: int = 60,
        tv_includes_half_factor: bool = True,
        tv_estimator: str = "tanh",
        r_max: float = 1.0,
        eps_tail: float = 0.0,
        bound_form: str = "linear",
        pilot_branch_factor: Optional[int] = None,
        likelihood_samples_per_distribution: int = 1,
        invalid_support_policy: str = "error",
        strict_vdra: bool = True,
        terminal_reward_fn: Optional[Any] = None,
    ):
        if mode == "hierachical":
            mode = "hierarchical"
        if mode not in {"subnode", "hierarchical", "perplexity"}:
            raise ValueError(f"Unsupported TV estimator mode: {mode}")
        if tv_estimator not in {"tanh", "legacy_abs"}:
            raise ValueError(f"Unsupported tv_estimator: {tv_estimator}")
        if invalid_support_policy not in {"error", "exclude", "resample"}:
            raise ValueError(f"Unsupported invalid_support_policy: {invalid_support_policy}")
        if strict_vdra and invalid_support_policy != "error":
            raise ValueError("strict_vdra requires invalid_support_policy='error'")
        if scorer is None and strict_vdra:
            raise ValueError("VDRA requires a conditional likelihood scorer")
        self.scorer = scorer
        self.node_expander = node_expander
        self.gamma = float(gamma)
        self.mode = mode
        if pilot_branch_factor is None:
            if n_tv_estimates is None:
                pilot_branch_factor = 8
            else:
                pilot_branch_factor = int(n_tv_estimates)
        self.pilot_branch_factor = max(int(pilot_branch_factor), 2)
        self.likelihood_samples_per_distribution = max(
            int(likelihood_samples_per_distribution), 1
        )
        self.total_support_samples = (
            self.pilot_branch_factor * self.likelihood_samples_per_distribution
        )
        if n_tv_estimates is not None and int(n_tv_estimates) != self.total_support_samples:
            raise ValueError(
                "n_tv_estimates is a deprecated alias for "
                "pilot_branch_factor * likelihood_samples_per_distribution; "
                f"got n_tv_estimates={int(n_tv_estimates)} but "
                f"{self.pilot_branch_factor} * "
                f"{self.likelihood_samples_per_distribution} = "
                f"{self.total_support_samples}"
            )
        self.n_tv_estimates = self.total_support_samples
        self.first_phase_tokens = max(int(first_phase_tokens), 1)
        self.second_phase_tokens = max(int(second_phase_tokens), 1)
        self.tv_includes_half_factor = bool(tv_includes_half_factor)
        self.tv_estimator = tv_estimator
        self.r_max = float(r_max)
        self.eps_tail = float(eps_tail)
        self.bound_form = bound_form
        self._score_cache: Dict[Tuple[Any, ...], float] = {}
        self.invalid_support_policy = invalid_support_policy
        self.strict_vdra = bool(strict_vdra)
        # P0.8: optional grader used to stamp terminal pilots with an observed
        # reward before dispersion is computed. When present, it is called on
        # each shortcut pilot node dict as terminal_reward_fn(node) -> float;
        # if it returns a number the node's "reward" field is set.
        self.terminal_reward_fn = terminal_reward_fn

    def _reward_variance(self, pair_tvs: Mapping[PairKey, float], n: int) -> float:
        return reward_variance_from_pair_tvs(
            pair_tvs,
            n=n,
            gamma=self.gamma,
            r_max=self.r_max,
            eps_tail=self.eps_tail,
            bound_form=self.bound_form,
        )

    async def estimate_for_parent(self, parent: Node, *, depth: int) -> TVEstimateResult:
        samples, candidates = await self._generate_samples(parent, depth=depth)
        result = await self.estimate_from_samples(samples)
        result.candidates = candidates
        return result


    def _terminal_augmented_dispersion(
        self,
        *,
        first_nodes: Sequence[Node],
        continuable_pairs: Sequence[Tuple[int, Node]],
        shortcut_nodes: Sequence[Node],
        local_pair_tvs: Mapping[PairKey, float],
    ) -> Tuple[Dict[PairKey, float], float, float, float, float]:
        """Return full-pilot TV pairs and component C values.

        Terminal phase-one pilots are complete distribution atoms. Without
        reward access in the scorer, terminal-vs-continuation support is a
        conservative disjoint atom (TV=1). P0.8: when terminal pilots have
        been graded (a numeric ``reward`` field is present on the shortcut
        node), the terminal-terminal contribution to the dispersion is
        ``(R_i - R_j)^2`` per pair rather than zero, matching:

            C_s = (1/n^2) * (Σ_cc B_ij² + Σ_tc B_ij² + Σ_tt (R_i-R_j)²).

        If no reward is available we keep terminal-terminal at 0 as a
        documented conservative bound.
        """

        n_total = len(first_nodes)
        if n_total < 2:
            return {}, 0.0, 0.0, 0.0, 0.0
        # P0.8: grade terminal pilots (no-op when no grader configured or
        # when a reward is already present).
        grader = getattr(self, "terminal_reward_fn", None)
        if grader is not None:
            for node in shortcut_nodes:
                if node.get("reward") is not None:
                    continue
                try:
                    reward = grader(node)
                except Exception:
                    reward = None
                if reward is not None:
                    node["reward"] = float(reward)
        global_pairs: Dict[PairKey, float] = {}
        local_to_global = {local: int(global_idx) for local, (global_idx, _) in enumerate(continuable_pairs)}
        for (i, j), tv in local_pair_tvs.items():
            if i in local_to_global and j in local_to_global:
                gi = local_to_global[i]
                gj = local_to_global[j]
                global_pairs[(min(gi, gj), max(gi, gj))] = float(tv)

        continuable_C = (
            self._reward_variance(local_pair_tvs, len(continuable_pairs))
            if len(continuable_pairs) >= 2 and local_pair_tvs
            else 0.0
        )
        shortcut_indices = [
            idx for idx, node in enumerate(first_nodes) if node in shortcut_nodes
        ]
        continuable_indices = {idx for idx, _ in continuable_pairs}
        cross_pairs: Dict[PairKey, float] = {}
        for ti in shortcut_indices:
            for ci in continuable_indices:
                pair = (min(ti, ci), max(ti, ci))
                cross_pairs[pair] = 1.0
                global_pairs[pair] = 1.0
        # P0.8: terminal-terminal contribution from observed rewards. Only
        # pairs where BOTH nodes have a numeric reward contribute; the rest
        # stay at 0 as documented above. We compute this Σ_tt (R_i-R_j)²/n²
        # directly rather than routing through _reward_variance because that
        # helper applies value_gap_bound() (a TV→value-gap transform) that is
        # correct for TV pairs but not for raw reward differences.
        terminal_pair_squared: Dict[PairKey, float] = {}
        for pos_i in range(len(shortcut_indices)):
            for pos_j in range(pos_i + 1, len(shortcut_indices)):
                gi = shortcut_indices[pos_i]
                gj = shortcut_indices[pos_j]
                ri = first_nodes[gi].get("reward")
                rj = first_nodes[gj].get("reward")
                if ri is None or rj is None:
                    continue
                try:
                    diff = float(ri) - float(rj)
                except (TypeError, ValueError):
                    continue
                terminal_pair_squared[(min(gi, gj), max(gi, gj))] = diff * diff
        terminal_C = (
            sum(terminal_pair_squared.values()) / float(n_total * n_total)
            if terminal_pair_squared
            else 0.0
        )
        cross_C = self._reward_variance(cross_pairs, n_total) if cross_pairs else 0.0
        # total_C = continuable_C + cross_C + terminal_C. continuable_C already
        # matches _reward_variance(local_pair_tvs, len(continuable_pairs));
        # cross_C uses TV=1 atoms; terminal_C is the observed reward variance
        # component computed above.
        total_C = self._reward_variance(global_pairs, n_total) if global_pairs else 0.0
        total_C = float(total_C) + float(terminal_C)
        return global_pairs, total_C, continuable_C, terminal_C, cross_C

    async def estimate_k_for_parent(
        self,
        parent: Node,
        *,
        depth: int,
        duplicate_tv_threshold: float,
    ) -> TVEstimateResult:
        """Estimate the number of distinct first-phase prefixes to expand.

        This is the online GEAR k-predictor path.  It keeps first-phase
        prefixes as candidate rollouts, scores every second-phase continuation
        under every first prefix, then clusters first prefixes whose pairwise TV
        is below ``duplicate_tv_threshold``.

        Pilots that terminate (EOS) inside the first phase are complete
        trajectories: they are excluded from the TV matrix entirely and
        returned as ``shortcut_candidates`` so the caller can attach them as
        graded leaf children (counted against the branch budget). Pair indices
        in ``pair_tvs`` therefore refer to the *continuable* pilot list.
        """

        first_nodes, support_nodes, support_origins = (
            await self._generate_first_prefix_support(parent, depth=depth)
        )
        if not first_nodes:
            return TVEstimateResult(
                samples=[],
                logp_matrix=[],
                prob_matrix=[],
                pair_tvs={},
                dispersion_C=0.0,
                candidates=[],
                predicted_k=0,
                unique_candidates=[],
            )

        shortcut_nodes = [
            node for node in first_nodes if node.get("finish_reason") != "length"
        ]
        continuable_pairs = [
            (idx, node)
            for idx, node in enumerate(first_nodes)
            if node.get("finish_reason") == "length"
        ]
        continuable_nodes = [node for _, node in continuable_pairs]
        local_index = {orig: local for local, (orig, _) in enumerate(continuable_pairs)}
        local_origins = [local_index[origin] for origin in support_origins]

        samples = [
            TVSample(first=first, second=second)
            for first in continuable_nodes
            for second in support_nodes
        ]
        if len(continuable_nodes) < 2 or not support_nodes:
            cluster_id_per_pilot = list(range(len(first_nodes)))
            representative_index_per_cluster = {idx: idx for idx in range(len(first_nodes))}
            cluster_size = {idx: 1 for idx in range(len(first_nodes))}
            cluster_members = {idx: [idx] for idx in range(len(first_nodes))}
            self._annotate_cluster_metadata(
                first_nodes,
                cluster_id_per_pilot=cluster_id_per_pilot,
                representative_index_per_cluster=representative_index_per_cluster,
                cluster_size=cluster_size,
                cluster_members=cluster_members,
            )
            global_pair_tvs, total_C, continuation_C, terminal_C, cross_C = (
                self._terminal_augmented_dispersion(
                    first_nodes=first_nodes,
                    continuable_pairs=continuable_pairs,
                    shortcut_nodes=shortcut_nodes,
                    local_pair_tvs={},
                )
            )
            return TVEstimateResult(
                samples=samples,
                logp_matrix=[],
                prob_matrix=[],
                pair_tvs=global_pair_tvs,
                dispersion_C=total_C,
                dispersion_C_continuation=continuation_C,
                dispersion_C_terminal=terminal_C,
                dispersion_C_cross=cross_C,
                candidates=first_nodes,
                predicted_k=len(shortcut_nodes) + len(continuable_nodes),
                unique_candidates=continuable_nodes,
                shortcut_candidates=shortcut_nodes,
                support_nodes=support_nodes,
                cluster_id_per_pilot=cluster_id_per_pilot,
                representative_index_per_cluster=representative_index_per_cluster,
                cluster_size=cluster_size,
            )

        logp_matrix = await self._score_matrix_from_nodes(continuable_nodes, support_nodes)
        prob_matrix = [self._support_probabilities(row) for row in logp_matrix]
        if self.tv_estimator == "tanh":
            pair_tvs = self._pair_tvs_tanh(logp_matrix, local_origins)
        else:
            pair_tvs = self._pair_tvs(prob_matrix)
        duplicate_pairs = [
            pair for pair, tv in pair_tvs.items() if tv < float(duplicate_tv_threshold)
        ]
        (
            local_cluster_ids,
            local_representatives,
            local_cluster_sizes,
            local_cluster_members,
        ) = self._cluster_prefixes(continuable_nodes, pair_tvs, duplicate_tv_threshold)

        cluster_id_per_pilot = [-1] * len(first_nodes)
        representative_index_per_cluster: Dict[int, int] = {}
        cluster_size: Dict[int, int] = {}
        cluster_members: Dict[int, List[int]] = {}
        for local_cluster_id, local_rep in local_representatives.items():
            global_cluster_id = int(local_cluster_id)
            global_rep = continuable_pairs[int(local_rep)][0]
            representative_index_per_cluster[global_cluster_id] = global_rep
            members = [
                continuable_pairs[int(local_idx)][0]
                for local_idx in local_cluster_members.get(local_cluster_id, [local_rep])
            ]
            cluster_size[global_cluster_id] = len(members)
            cluster_members[global_cluster_id] = members
        for local_idx, cluster_id in enumerate(local_cluster_ids):
            global_idx = continuable_pairs[local_idx][0]
            cluster_id_per_pilot[global_idx] = int(cluster_id)

        next_cluster_id = len(local_representatives)
        for shortcut_idx, _ in enumerate(shortcut_nodes):
            original_idx = first_nodes.index(shortcut_nodes[shortcut_idx])
            cluster_id = next_cluster_id
            next_cluster_id += 1
            cluster_id_per_pilot[original_idx] = cluster_id
            representative_index_per_cluster[cluster_id] = original_idx
            cluster_size[cluster_id] = 1
            cluster_members[cluster_id] = [original_idx]

        self._annotate_cluster_metadata(
            first_nodes,
            cluster_id_per_pilot=cluster_id_per_pilot,
            representative_index_per_cluster=representative_index_per_cluster,
            cluster_size=cluster_size,
            cluster_members=cluster_members,
        )
        unique_indices = sorted(
            representative_index_per_cluster[cid]
            for cid in local_representatives.keys()
        )
        unique_candidates = [first_nodes[idx] for idx in unique_indices]
        global_pair_tvs, total_C, continuation_C, terminal_C, cross_C = (
            self._terminal_augmented_dispersion(
                first_nodes=first_nodes,
                continuable_pairs=continuable_pairs,
                shortcut_nodes=shortcut_nodes,
                local_pair_tvs=pair_tvs,
            )
        )
        return TVEstimateResult(
            samples=samples,
            logp_matrix=logp_matrix,
            prob_matrix=prob_matrix,
            pair_tvs=global_pair_tvs,
            dispersion_C=total_C,
            dispersion_C_continuation=continuation_C,
            dispersion_C_terminal=terminal_C,
            dispersion_C_cross=cross_C,
            candidates=first_nodes,
            predicted_k=len(shortcut_nodes) + len(unique_candidates),
            unique_candidates=unique_candidates,
            duplicate_pairs=duplicate_pairs,
            shortcut_candidates=shortcut_nodes,
            support_nodes=support_nodes,
            cluster_id_per_pilot=cluster_id_per_pilot,
            representative_index_per_cluster=representative_index_per_cluster,
            cluster_size=cluster_size,
        )

    async def estimate_from_samples(self, samples: Sequence[TVSample]) -> TVEstimateResult:
        samples = list(samples)
        n = len(samples)
        if n < 2:
            return TVEstimateResult(
                samples=samples,
                logp_matrix=[],
                prob_matrix=[],
                pair_tvs={},
                dispersion_C=0.0,
            )

        logp_matrix = await self._score_matrix_from_nodes(
            [sample.first for sample in samples],
            [sample.second for sample in samples],
        )
        prob_matrix = [self._support_probabilities(row) for row in logp_matrix]
        if self.tv_estimator == "tanh":
            # Column k was generated from sample k's first prefix, so pair
            # (i, j) restricts to its own continuations Z_i ∪ Z_j.
            pair_tvs = self._pair_tvs_tanh(logp_matrix, list(range(n)))
        else:
            pair_tvs = self._pair_tvs(prob_matrix)
        variance = self._reward_variance(pair_tvs, n)
        return TVEstimateResult(
            samples=samples,
            logp_matrix=logp_matrix,
            prob_matrix=prob_matrix,
            pair_tvs=pair_tvs,
            dispersion_C=variance,
        )

    async def _generate_samples(
        self, parent: Node, *, depth: int
    ) -> Tuple[List[TVSample], List[Node]]:
        if self.mode == "hierarchical":
            first_count = self.pilot_branch_factor
            second_per_first = self.likelihood_samples_per_distribution
        elif self.mode == "perplexity":
            first_count = self._perplexity_branch_factor(
                parent, fallback=self._hierarchical_first_count()
            )
            second_per_first = None
        else:
            first_count = self.pilot_branch_factor
            second_per_first = self.likelihood_samples_per_distribution

        first_nodes = await self._expand(
            current_node=parent,
            prefix=parent.get("full_text", ""),
            depth=depth,
            max_tokens=self.first_phase_tokens,
            branch_factor=first_count,
        )

        # A non-`length` finish reason means the model reached a terminal
        # response during phase one.  Keep it as a reusable budget candidate,
        # but do not ask the model to continue from an already-finished prefix.
        continuable_first_nodes = [
            first for first in first_nodes if first.get("finish_reason") == "length"
        ]

        samples: List[TVSample] = []
        second_tasks = [
            asyncio.create_task(
                self._expand(
                    current_node=first,
                    prefix=first.get("full_text", ""),
                    depth=depth + 1,
                    max_tokens=self.second_phase_tokens,
                    branch_factor=(
                        self._perplexity_branch_factor(first, fallback=2)
                        if self.mode == "perplexity"
                        else second_per_first
                    ),
                )
            )
            for first in continuable_first_nodes
        ]
        second_batches = await asyncio.gather(*second_tasks) if second_tasks else []
        for first, seconds in zip(continuable_first_nodes, second_batches):
            for second in seconds:
                samples.append(TVSample(first=first, second=second))

        return samples, first_nodes


    async def _generate_first_prefix_support(
        self, parent: Node, *, depth: int
    ) -> Tuple[List[Node], List[Node], List[int]]:
        if self.mode == "hierarchical":
            first_count = self.pilot_branch_factor
            second_per_first = self.likelihood_samples_per_distribution
        elif self.mode == "perplexity":
            first_count = self._perplexity_branch_factor(
                parent, fallback=self._hierarchical_first_count()
            )
            second_per_first = None
        else:
            first_count = self.pilot_branch_factor
            second_per_first = self.likelihood_samples_per_distribution

        first_nodes = await self._expand(
            current_node=parent,
            prefix=parent.get("full_text", ""),
            depth=depth,
            max_tokens=self.first_phase_tokens,
            branch_factor=first_count,
        )
        continuable = [
            (idx, first)
            for idx, first in enumerate(first_nodes)
            if first.get("finish_reason") == "length"
        ]
        second_tasks = [
            asyncio.create_task(
                self._expand(
                    current_node=first,
                    prefix=first.get("full_text", ""),
                    depth=depth + 1,
                    max_tokens=self.second_phase_tokens,
                    branch_factor=(
                        self._perplexity_branch_factor(first, fallback=2)
                        if self.mode == "perplexity"
                        else second_per_first
                    ),
                )
            )
            for _, first in continuable
        ]
        second_batches = await asyncio.gather(*second_tasks) if second_tasks else []
        support_nodes: List[Node] = []
        # support_origins[k] = index (in first_nodes) of the prefix that
        # generated support block k; the tanh estimator uses it to restrict
        # each pair (i, j) to its own continuations Z_i ∪ Z_j.
        support_origins: List[int] = []

        for (first_idx, _), seconds in zip(continuable, second_batches):
            for second in seconds:

                support_nodes.append(second)
                support_origins.append(first_idx)
        return first_nodes, support_nodes, support_origins

    @staticmethod
    def _cluster_prefixes(
        first_nodes: Sequence[Node],
        pair_tvs: Mapping[PairKey, float],
        duplicate_tv_threshold: float,
    ) -> Tuple[List[int], Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
        """Cluster duplicate pilots deterministically without reward access.

        P0.7: uses star clustering around a deterministic representative so
        every cluster satisfies max_{x in G_r} TV(x, r) <= eps by
        construction. Connected components (used previously) only enforce
        pairwise-near paths and admit clusters where two members can be
        arbitrarily far apart via an intermediate.
        """

        n = len(first_nodes)

        def _tv(i: int, j: int) -> float:
            if i == j:
                return 0.0
            key = (i, j) if i < j else (j, i)
            return float(pair_tvs.get(key, float("inf")))

        threshold = float(duplicate_tv_threshold)
        cluster_id_per_pilot: List[int] = [-1] * n
        representative_index_per_cluster: Dict[int, int] = {}
        cluster_members: Dict[int, List[int]] = {}
        # Deterministic order = original pilot index. Every pilot either starts
        # a new cluster (becoming its representative) or joins the lowest-id
        # existing cluster whose representative is within eps.
        for idx in range(n):
            best_cluster: Optional[int] = None
            for cluster_id, rep in representative_index_per_cluster.items():
                if _tv(idx, rep) < threshold:
                    if best_cluster is None or cluster_id < best_cluster:
                        best_cluster = cluster_id
            if best_cluster is None:
                new_id = len(representative_index_per_cluster)
                representative_index_per_cluster[new_id] = idx
                cluster_members[new_id] = [idx]
                cluster_id_per_pilot[idx] = new_id
            else:
                cluster_members[best_cluster].append(idx)
                cluster_id_per_pilot[idx] = best_cluster

        cluster_size: Dict[int, int] = {
            cid: len(members) for cid, members in cluster_members.items()
        }
        return cluster_id_per_pilot, representative_index_per_cluster, cluster_size, cluster_members

    @staticmethod
    def _annotate_cluster_metadata(
        first_nodes: Sequence[Node],
        *,
        cluster_id_per_pilot: Sequence[int],
        representative_index_per_cluster: Mapping[int, int],
        cluster_size: Mapping[int, int],
        cluster_members: Mapping[int, Sequence[int]],
    ) -> None:
        total = max(len(first_nodes), 1)
        representative_for_index = {
            int(rep): int(cluster_id)
            for cluster_id, rep in representative_index_per_cluster.items()
        }
        for idx, node in enumerate(first_nodes):
            cluster_id = int(cluster_id_per_pilot[idx])
            members = list(cluster_members.get(cluster_id, [idx]))
            multiplicity = int(cluster_size.get(cluster_id, len(members)))
            node["vdra_cluster_id"] = cluster_id
            node["vdra_cluster_multiplicity"] = multiplicity
            node["vdra_original_pilot_indices"] = members
            if idx in representative_for_index:
                node["vdra_representative_weight"] = multiplicity / float(total)

    @staticmethod
    def _unique_prefix_indices(
        first_nodes: Sequence[Node],
        pair_tvs: Mapping[PairKey, float],
        duplicate_tv_threshold: float,
    ) -> List[int]:
        """Prune duplicates by duplicate-degree, not by generation score.

        Two prefixes with pairwise TV below ``duplicate_tv_threshold`` are
        duplicates. We repeatedly prune the prefix with the most duplicate
        partners (ties broken toward the larger index so earlier pilots
        survive) until no duplicate pair remains. Survivor selection carries
        no likelihood bias, so downstream reuse can sample uniformly.
        """

        n = len(first_nodes)
        adjacency: Dict[int, set] = {idx: set() for idx in range(n)}
        for (i, j), tv in pair_tvs.items():
            if i != j and tv < float(duplicate_tv_threshold):
                adjacency[i].add(j)
                adjacency[j].add(i)

        alive = set(range(n))
        while alive:
            degree = {idx: len(adjacency[idx] & alive) for idx in alive}
            worst = max(alive, key=lambda idx: (degree[idx], idx))
            if degree[worst] == 0:
                break
            alive.remove(worst)
        return sorted(alive)

    def _hierarchical_first_count(self) -> int:
        return max(2, int(math.ceil(math.sqrt(self.total_support_samples))))

    def _perplexity_branch_factor(self, node: Mapping[str, Any], *, fallback: int) -> int:
        perplexity = self._node_perplexity(node)
        if perplexity is None:
            return max(2, int(fallback))
        return max(2, int(math.ceil(perplexity)))

    @staticmethod
    def _node_perplexity(node: Mapping[str, Any]) -> Optional[float]:
        sum_logprobs = node.get("sum_logprobs")
        num_tokens = node.get("num_tokens")
        if sum_logprobs is None or num_tokens is None:
            return None
        try:
            token_count = int(num_tokens)
            if token_count <= 0:
                return None
            avg_negative_logprob = -float(sum_logprobs) / token_count
            perplexity = math.exp(avg_negative_logprob)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(perplexity):
            return None
        return perplexity

    async def _expand(
        self,
        *,
        current_node: Node,
        prefix: str,
        depth: int,
        max_tokens: int,
        branch_factor: int,
    ) -> List[Node]:
        try:
            return await self.node_expander.expand(
                current_node=current_node,
                prefix=prefix,
                depth=depth,
                max_tokens=max_tokens,
                branch_factor=branch_factor,
            )
        except TypeError:
            return await self.node_expander.expand(
                current_node=current_node,
                prefix=prefix,
                depth=depth,
                max_tokens=max_tokens,
            )

    @staticmethod
    def _token_ids(value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        try:
            return [int(tok) for tok in value]
        except (TypeError, ValueError):
            return None

    def _prefix_token_ids(self, node: Mapping[str, Any]) -> Optional[List[int]]:
        return self._token_ids(node.get("full_token_ids"))

    def _continuation_token_ids(self, node: Mapping[str, Any]) -> Optional[List[int]]:
        token_ids = self._token_ids(node.get("response_token_ids"))
        if token_ids is not None:
            return token_ids
        return self._token_ids(node.get("token_ids"))

    async def _score_one_any(self, prefix: str, continuation: str) -> float:
        """Await the legacy text scorer if it is async; accept sync scorers as-is."""

        result = self.scorer.score_one(prefix, continuation)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return await result
        return result

    async def _score_one_tokens_any(
        self,
        prefix_token_ids: Optional[Sequence[int]],
        continuation_token_ids: Optional[Sequence[int]],
        *,
        prefix_text: str,
        continuation_text: str,
    ) -> float:
        scorer_fn = getattr(self.scorer, "score_one_tokens", None)
        if scorer_fn is None:
            scorer_fn = getattr(self.scorer, "score_one_token_ids", None)
        if scorer_fn is not None and prefix_token_ids is not None and continuation_token_ids is not None:
            result = scorer_fn(list(prefix_token_ids), list(continuation_token_ids))
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                result = await result
            return result
        if self.strict_vdra:
            raise ValueError(
                "strict VDRA requires exact token-id scoring via score_one_tokens "
                "and nodes with full_token_ids/response_token_ids"
            )
        return await self._score_one_any(prefix_text, continuation_text)

    async def _score_matrix_from_nodes(
        self,
        prefix_nodes: Sequence[Node],
        support_nodes: Sequence[Node],
    ) -> List[List[float]]:
        matrix: List[List[float]] = [[0.0 for _ in support_nodes] for _ in prefix_nodes]
        pending: List[Tuple[int, int, Tuple[Any, ...], asyncio.Task]] = []
        for i, prefix_node in enumerate(prefix_nodes):
            prefix_tokens = self._prefix_token_ids(prefix_node)
            prefix_text = str(prefix_node.get("full_text", ""))
            for k, support_node in enumerate(support_nodes):
                continuation_tokens = self._continuation_token_ids(support_node)
                continuation_text = str(support_node.get("text", ""))
                if prefix_tokens is not None and continuation_tokens is not None:
                    key: Tuple[Any, ...] = (
                        "tokens", tuple(prefix_tokens), tuple(continuation_tokens)
                    )
                else:
                    key = ("text", prefix_text, continuation_text)
                if key in self._score_cache:
                    matrix[i][k] = self._score_cache[key]
                else:
                    task = asyncio.create_task(
                        self._score_one_tokens_any(
                            prefix_tokens,
                            continuation_tokens,
                            prefix_text=prefix_text,
                            continuation_text=continuation_text,
                        )
                    )
                    pending.append((i, k, key, task))
        if pending:
            values = await asyncio.gather(*(task for _, _, _, task in pending))
            for (i, k, key, _), value in zip(pending, values):
                value = float(value)
                self._score_cache[key] = value
                matrix[i][k] = value
        return matrix

    async def _score_matrix(
        self,
        prefixes: Sequence[str],
        support: Sequence[str],
    ) -> List[List[float]]:
        if self.strict_vdra:
            raise ValueError("strict VDRA requires node-based exact token-id scoring")
        matrix: List[List[float]] = [[0.0 for _ in support] for _ in prefixes]
        pending: List[Tuple[int, int, Tuple[Any, ...], asyncio.Task]] = []
        for i, prefix in enumerate(prefixes):
            for k, continuation in enumerate(support):
                key: Tuple[Any, ...] = ("text", prefix, continuation)
                if key in self._score_cache:
                    matrix[i][k] = self._score_cache[key]
                else:
                    pending.append((i, k, key, asyncio.create_task(self._score_one_any(prefix, continuation))))
        if pending:
            values = await asyncio.gather(*(task for _, _, _, task in pending))
            for (i, k, key, _), value in zip(pending, values):
                value = float(value)
                self._score_cache[key] = value
                matrix[i][k] = value
        return matrix

    @staticmethod
    def _support_probabilities(logps: Sequence[float]) -> List[float]:
        probs: List[float] = []
        for logp in logps:
            try:
                value = float(logp)
            except (TypeError, ValueError):
                probs.append(0.0)
                continue
            if not math.isfinite(value):
                probs.append(0.0 if value < 0.0 else 1.0)
                continue
            probs.append(math.exp(min(value, 0.0)))
        return probs

    def _pair_tvs(self, prob_matrix: Sequence[Sequence[float]]) -> Dict[PairKey, float]:
        """Legacy absolute-probability estimator (``tv_estimator='legacy_abs'``).

        Kept only for ablations: summing |exp(LP_i) - exp(LP_j)| over a sampled
        support of full sequences is numerically degenerate (sequence
        probabilities underflow toward 0), so it systematically reports TV ≈ 0.
        """

        pair_tvs: Dict[PairKey, float] = {}
        for i in range(len(prob_matrix)):
            pi = prob_matrix[i]
            for j in range(i + 1, len(prob_matrix)):
                pj = prob_matrix[j]
                tv = sum(abs(a - b) for a, b in zip(pi, pj))
                if self.tv_includes_half_factor:
                    tv *= 0.5
                pair_tvs[(i, j)] = tv
        return pair_tvs

    def _pair_tvs_tanh(
        self,
        logp_matrix: Sequence[Sequence[float]],
        support_origins: Optional[Sequence[int]] = None,
    ) -> Dict[PairKey, float]:
        """Pairwise TV via the §9 tanh estimator on the log-prob matrix.

        Pair (i, j) is estimated only on its own mixture support. Samples from
        unrelated distributions are never used as a fallback.
        """

        pair_tvs: Dict[PairKey, float] = {}
        n = len(logp_matrix)
        n_cols = len(logp_matrix[0]) if n else 0
        for i in range(n):
            for j in range(i + 1, n):
                cols: List[int] = []
                if support_origins is not None:
                    cols = [
                        k
                        for k, origin in enumerate(support_origins)
                        if origin in (i, j) and k < n_cols
                    ]
                if not cols:
                    if self.invalid_support_policy == "exclude":
                        continue
                    if self.invalid_support_policy == "resample":
                        raise RuntimeError(f"Pair ({i}, {j}) requires support resampling")
                    raise ValueError(f"Pair ({i}, {j}) has no pair-specific TV support")
                tv, valid_count = _pairwise_tv_tanh_with_count(
                    [logp_matrix[i][k] for k in cols],
                    [logp_matrix[j][k] for k in cols],
                )
                if valid_count == 0:
                    if self.invalid_support_policy == "exclude":
                        continue
                    if self.invalid_support_policy == "resample":
                        raise RuntimeError(f"Pair ({i}, {j}) requires support resampling")
                    raise ValueError(
                        "No valid pair-specific likelihood-ratio samples for TV estimation"
                    )
                pair_tvs[(i, j)] = tv
        return pair_tvs


def select_reuse_candidates(
    candidates: Sequence[Node], count: int, *, seed: Any
) -> List[Node]:
    """Uniformly sample ``count`` reuse pilots from the post-pruning survivors.

    Reuse selection must not prefer high-likelihood pilots (that would bias the
    child sample the node value is estimated from), so the only allowed
    selection rule is a seeded uniform draw over the surviving candidates.
    """

    pool = list(candidates)
    count = max(int(count), 0)
    if count >= len(pool):
        return pool
    return random.Random(seed).sample(pool, count)


def _candidate_score(node: Mapping[str, Any]) -> float:
    value = node.get("sum_logprobs")
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def rank_samples_by_score(samples: Sequence[TVSample]) -> List[TVSample]:
    """Return samples sorted by descending first-phase generation score."""

    def score(sample: TVSample) -> float:
        first = sample.first
        if first.get("sum_logprobs") is not None:
            return float(first["sum_logprobs"])
        return 0.0

    return sorted(samples, key=score, reverse=True)
