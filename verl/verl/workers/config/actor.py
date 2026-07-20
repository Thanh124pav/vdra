# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg',
            'treetune_ppo', 'vdra_segment_mean_ppo', 'vdra_node_balanced_ppo'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        segment_token_reduction (str): Within-segment token reduction for the
            VDRA segment-mean loss (PLAN.md P0.1). Must be exactly ``"mean"`` or
            ``"sum"``. ``mean`` is the canonical main-run default; ``sum`` is a
            supported first-class ablation. Any other value fails at startup.
        use_prob_mask (bool): Apply the treetune-style probability mask on top of
            the response mask before the PPO surrogate reduction (VDRA path).
            BOTH values are first-class supported; strict mode never forces
            one (PLAN.md §1.3).
        probability_mask_threshold (float): Authoritative threshold for the
            probability mask. A response token is active iff
            ``exp(old_log_prob) < probability_mask_threshold`` (STRICT ``<``).
            Must satisfy ``0 < t <= 1``; the historical default is ``0.9``.
            Extraction-time active-token counting and the actor-side mask read
            this same field — the threshold is never hard-coded.
        ratio_threshold (float): Diagnostic ratio threshold for VDRA losses.
            ``float('inf')`` disables the report (the canonical VDRA main path
            never uses this to drop microbatches — see PLAN.md P0.4).
        policy_aggregation (str): Objective aggregation for the VDRA
            segment-mean loss (PLAN.md §1.3, user decision 2026-07-21):
            ``"segment_mean"`` — every original logical segment slot has equal
            weight ``1/M_B`` (pre-filter logical-batch slot count);
            ``"token_mean"`` — every original valid token has equal weight
            ``1/T_B`` (pre-filter logical-batch token count);
            ``"tree_balanced_segment_mean"`` — labeled ABLATION: the
            historical ``w = 1/(N_T * N_seg(T))`` tree-balanced objective.
            The retired name ``"global_segment_mean"`` fails fast with a
            rename error (it must never silently mean the new uniform
            ``segment_mean``). The former ``batch_slot_mean_ablation`` flag
            was retired: its mathematics IS ``segment_mean``.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    segment_token_reduction: str = "mean"
    use_prob_mask: bool = True
    probability_mask_threshold: float = 0.9
    ratio_threshold: float = float("inf")
    # PLAN.md §1.3 (2026-07-21): the canonical default is the paper's
    # uniform segment mean over the pre-filter logical-batch slot count.
    policy_aggregation: str = "segment_mean"

    _VALID_POLICY_AGGREGATIONS = ("token_mean", "segment_mean", "tree_balanced_segment_mean")
    # PLAN.md §1.3: the retired name is accepted by TYPED PARSING only so the
    # strictness-aware cross-level validator can decide its fate (strict:
    # fail fast with the rename message; non-strict: canonicalize to
    # tree_balanced_segment_mean with a DeprecationWarning). It must never
    # reach runtime production code.
    LEGACY_POLICY_AGGREGATION = "global_segment_mean"

    def __post_init__(self):
        """PLAN.md P0.1/§1.3: fail loudly on invalid VDRA loss knobs.

        Only ``mean`` and ``sum`` are accepted for
        ``segment_token_reduction`` (see PLAN.md §1.2 and P0.1), and only the
        three explicit aggregation names for ``policy_aggregation`` (plus the
        legacy ``global_segment_mean`` token, which survives parsing and is
        resolved by cross-level validation). We normalise to lowercase so
        YAML overrides like ``Mean`` do not silently fall back to the default.
        """
        raw = self.segment_token_reduction
        if raw is None:
            raw = "mean"
        normalised = str(raw).strip().lower()
        if normalised not in {"mean", "sum"}:
            raise ValueError(
                f"segment_token_reduction={raw!r} is invalid; must be exactly "
                "one of {'mean', 'sum'} (PLAN.md P0.1)."
            )
        # Field is a normal string on this frozen-ish dataclass wrapper; use
        # object.__setattr__ so BaseConfig's mutability guards do not fight us.
        object.__setattr__(self, "segment_token_reduction", normalised)

        raw_agg = self.policy_aggregation
        if raw_agg is None:
            # PLAN.md §12: ONE canonical default everywhere.
            raw_agg = "segment_mean"
        agg = str(raw_agg).strip().lower()
        if agg not in self._VALID_POLICY_AGGREGATIONS + (
            self.LEGACY_POLICY_AGGREGATION,
        ):
            raise ValueError(
                f"policy_aggregation={raw_agg!r} is invalid; must be exactly "
                f"one of {self._VALID_POLICY_AGGREGATIONS} (PLAN.md §1.3)."
            )
        if agg == "token_mean" and normalised == "sum":
            raise ValueError(
                "policy_aggregation='token_mean' has no within-segment "
                "reduction; segment_token_reduction='sum' would be silently "
                "ignored, which strict VDRA forbids (PLAN.md §1.3). Leave "
                "segment_token_reduction='mean'."
            )
        object.__setattr__(self, "policy_aggregation", agg)

        # PLAN.md §1: authoritative probability-mask threshold.
        raw_thr = self.probability_mask_threshold
        if raw_thr is None:
            raw_thr = 0.9
        try:
            threshold = float(raw_thr)
        except (TypeError, ValueError):
            raise ValueError(
                f"probability_mask_threshold={raw_thr!r} is not a number "
                "(PLAN.md §1)."
            ) from None
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"probability_mask_threshold={raw_thr!r} is invalid; must "
                "satisfy 0 < threshold <= 1 (PLAN.md §1)."
            )
        object.__setattr__(self, "probability_mask_threshold", threshold)


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    rollout_n: int = 1  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
