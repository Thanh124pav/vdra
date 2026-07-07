// Reusable overlay that converts an SPO-tree config into an GEAR-tree
// config. Apply via:
//
//   (import 'polIter_<model>_<dataset>_spo_tree.jsonnet')
//   + (import 'gear_defaults.libsonnet')
//   + (import 'gear_overlay.libsonnet')
//
// The overlay reads `$.gear.*` (provided by gear_defaults.libsonnet)
// and mirrors the knobs onto the inference strategy / episode generator
// where SPO's main entrypoint will pass them to GEAR's __init__.
//
// Anything specific to a single experiment (k algorithm, generation mode, budget, ...)
// is supplied by ablation snippets that override `$.gear.*` BEFORE this
// overlay is applied. Because Jsonnet evaluates everything lazily, the
// final values come out the way the caller chained them.
{
  episode_generator+: {
    type: 'gear_episode_generator',

    inference_strategy+: {
      type: 'gear',
      node_expander+: {
        program_kwargs+: {
          logprobs: 1,
        },
      },
      gear_epsilon: $.gear.epsilon,
      gear_r_max: $.gear.r_max,
      gear_gamma: $.gear.gamma,
      gear_score_concurrency: $.gear.score_concurrency,
      gear_score_timeout_seconds: $.gear.score_timeout_seconds,
      gear_score_retry_attempts: $.gear.score_retry_attempts,
      gear_score_retry_backoff_seconds: $.gear.score_retry_backoff_seconds,
      gear_k_algorithm: $.gear.k_algorithm,
      gear_generation_mode: $.gear.generation_mode,
      gear_n_tv_estimates: $.gear.n_tv_estimates,
      gear_tv_subnode_max_tokens: $.gear.tv_subnode_max_tokens,
      gear_tv_second_phase_tokens: $.gear.tv_second_phase_tokens,
      gear_tv_includes_half_factor: $.gear.tv_includes_half_factor,
      gear_budget_lambda: $.gear.budget_lambda,
      gear_n_min: $.gear.n_min,
      gear_budget_overhead_mode: $.gear.budget_overhead_mode,
      gear_allocation_mode: $.gear.allocation_mode,
      gear_use_residual_budget: $.gear.use_residual_budget,
      gear_budget_queue_count: $.gear.budget_queue_count,
      gear_budget_queue_timeout_seconds: $.gear.budget_queue_timeout_seconds,
      gear_skip_near_leaf_expand: $.gear.skip_near_leaf_expand,
      gear_root_allocation: $.gear.root_allocation,
    },

    tree_update_mode: $.gear.tree_update_mode,
    treepo_global_weight: $.gear.treepo_global_weight,
    treerl_gamma: $.gear.treerl_gamma,
    gear_zero_advantage_when_pruned: $.gear.zero_advantage_when_pruned,
    gear_emit_pruned_edges: $.gear.emit_pruned_edges,
    gear_demo_examples_per_tree: $.gear.demo_examples_per_tree,
    gear_demos_dir: $.gear.demos_dir,
    gear_log_demos_to_wandb: $.gear.log_demos_to_wandb,
    gear_full_tree_demo_every_n_trees: $.gear.full_tree_demo_every_n_trees,
    gear_full_tree_demo_max_trees: $.gear.full_tree_demo_max_trees,
    gear_tree_policy_algorithm_name: $.gear.tree_policy_algorithm_name,
    gear_tree_policy_segmentation_type: $.gear.tree_policy_segmentation_type,
    gear_tree_policy_tree_shape: $.gear.tree_policy_tree_shape,
    gear_tree_policy_tree_m: $.gear.tree_policy_tree_m,
    gear_print_run_manifest: $.gear.print_run_manifest,
  },
}
