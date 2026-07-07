"""GEAR algorithm components.

The package initializer intentionally avoids eager submodule imports so that
analysis-only utilities can load lightweight helpers without pulling optional
runtime dependencies such as numpy, transformers, or vLLM clients.
"""

__all__ = [
    "budget_allocation",
    "local_value_share",
    "log_prob_matrix",
    "logging_helpers",
    "lp_scorer",
    "pruning_controller",
    "segment_index",
    "thresholds",
    "tree_policy_logging",
    "triggers",
    "tv_distance",
    "vllm_scorer",
]
