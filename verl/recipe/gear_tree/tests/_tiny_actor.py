"""Shared tiny-model + REAL-actor builders for gear_tree actor tests.

Test infrastructure only (never imported by production code). Extracted from
``test_actor_update_control_flow.py`` so the FSDP2 parity harness and the
control-flow test drive the exact same production entry points:

* :class:`TinyLM` — minimal causal-LM-shaped module. It is compatible with
  the REAL ``verl.utils.fsdp_utils.apply_fsdp2`` wrapper: ``fully_shard``
  wraps :class:`TinyBlock` (via ``_no_split_modules``) and the root module,
  and ``config.tie_word_embeddings=True`` keeps the embedding on the root
  wrap exactly like a tied-embedding HF model.
* :func:`make_edges` — canonical replay-edge dicts with configurable tree
  sizes, deterministic uneven response lengths, and signed non-zero
  advantages (row order ``contiguous`` or ``interleaved``).
* :func:`build_batch` — tensorization through the REAL ``edges_to_dataproto``
  with stored old log-probs forced, as the replay path does.
* :func:`make_actor_config` / :func:`make_actor` — typed ``FSDPActorConfig``
  and a REAL ``DataParallelPPOActor`` (no mirror reimplementation).
"""

from __future__ import annotations

from types import SimpleNamespace

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat import when mp.spawn re-imports test modules
    import _test_shims

_test_shims.install()

import torch

import recipe.gear_tree.policy_loss  # noqa: F401 — registers the VDRA losses
from recipe.gear_tree.tree_data import edges_to_dataproto

VOCAB = 32
MAX_PROMPT = 6
MAX_RESPONSE = 4


class Tok:
    pad_token_id = 0
    eos_token_id = 0


class TinyBlock(torch.nn.Module):
    """Wrap target for ``apply_fsdp2`` (listed in ``_no_split_modules``)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class TinyLM(torch.nn.Module):
    """Minimal causal-LM-shaped module: returns ``.logits`` like HF models.

    ``torch.manual_seed(0)`` in ``__init__`` makes every instance identical,
    so distributed ranks and single-rank references start from the same
    weights without any state_dict broadcast.
    """

    _no_split_modules = ["TinyBlock"]

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embed = torch.nn.Embedding(VOCAB, 8)
        self.block = TinyBlock()
        self.head = torch.nn.Linear(8, VOCAB)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=False, **_):
        logits = self.head(self.block(self.embed(input_ids)))
        return SimpleNamespace(logits=logits)


def make_edges(
    tree_sizes: list[int],
    *,
    order: str = "contiguous",
    advantage_for=None,
    resp_len_for=None,
) -> list[dict]:
    """Canonical replay-edge dicts for ``len(tree_sizes)`` trees.

    Tree ``t`` contributes ``tree_sizes[t]`` edges with the PRE-FILTER
    ``tree_total_segment_count = tree_sizes[t]``. Defaults produce uneven
    response lengths (1..MAX_RESPONSE) and signed non-zero advantages, both
    deterministic in the global emission index.

    ``order="contiguous"`` emits tree by tree (production reservation order);
    ``order="interleaved"`` round-robins across trees so every consecutive
    row block mixes as many distinct trees as possible.
    """
    if order not in ("contiguous", "interleaved"):
        raise ValueError(f"unknown edge order: {order!r}")

    per_tree: list[list[dict]] = []
    i = 0  # global emission index driving all deterministic variation
    for t, size in enumerate(tree_sizes):
        rows = []
        for j in range(size):
            rl = resp_len_for(t, j) if resp_len_for else 1 + (i % MAX_RESPONSE)
            adv = (
                advantage_for(t, j)
                if advantage_for
                else ((-1.0) ** i) * (0.3 + 0.1 * (i % 5))
            )
            rows.append(
                {
                    "edge_id": f"t{t}/e{j}",
                    "tree_id": f"t{t}",
                    "parent_group_id": f"t{t}/pg",
                    "child_segment_id": f"t{t}/e{j}",
                    "question_id": f"q{t // 4}",
                    "allocated_k": size,
                    "sample_multiplicity": 1,
                    "tree_total_segment_count": size,
                    "queue_flush_id": "0",
                    "queue_released_segment_count": size,
                    "query_token_ids": [1, 2 + (i % 5)],
                    "response_token_ids": [1 + ((i + k) % (VOCAB - 1)) for k in range(rl)],
                    "actor_shifted_log_probs": [-0.5 + 0.01 * ((i + k) % 7) for k in range(rl)],
                    "advantage": adv,
                    "value": 0.4,
                    "reward": 1.0,
                }
            )
            i += 1
        per_tree.append(rows)

    if order == "contiguous":
        return [row for rows in per_tree for row in rows]

    interleaved: list[dict] = []
    depth = 0
    while any(depth < len(rows) for rows in per_tree):
        for rows in per_tree:
            if depth < len(rows):
                interleaved.append(rows[depth])
        depth += 1
    return interleaved


def build_batch(edges: list[dict]):
    """Tensorize edges through the REAL production ``edges_to_dataproto``."""
    batch = edges_to_dataproto(
        edges,
        Tok(),
        max_prompt_length=MAX_PROMPT,
        max_response_length=MAX_RESPONSE,
        include_old_log_probs=True,
        loss_mode="vdra_segment_mean_ppo",
    )
    batch.meta_info["temperature"] = 1.0
    # P0.4: replay edges force the stored PPO denominator.
    batch.meta_info["force_stored_old_log_probs"] = True
    return batch


def make_actor_config(
    *,
    strategy: str = "fsdp",
    mini: int = 128,
    micro: int = 64,
    reduction: str = "mean",
    aggregation: str | None = None,
    grad_clip: float = 1.0,
):
    from verl.workers.config.actor import FSDPActorConfig, PolicyLossConfig

    policy_loss_kwargs = dict(
        loss_mode="vdra_segment_mean_ppo",
        segment_token_reduction=reduction,
        use_prob_mask=False,
    )
    if aggregation is not None:
        policy_loss_kwargs["policy_aggregation"] = aggregation
    return FSDPActorConfig(
        strategy=strategy,
        rollout_n=1,
        ppo_mini_batch_size=mini,
        ppo_micro_batch_size_per_gpu=micro,
        ppo_epochs=1,
        clip_ratio=0.2,
        grad_clip=grad_clip,
        use_torch_compile=False,
        use_dynamic_bsz=False,
        policy_loss=PolicyLossConfig(**policy_loss_kwargs),
    )


def make_actor(model: torch.nn.Module | None = None, config=None, lr: float = 0.05):
    """REAL ``DataParallelPPOActor`` over ``model`` (default: fresh TinyLM)."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyLM().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    actor = DataParallelPPOActor(
        config=config if config is not None else make_actor_config(),
        actor_module=model,
        actor_optimizer=optimizer,
    )
    return actor, model, optimizer
