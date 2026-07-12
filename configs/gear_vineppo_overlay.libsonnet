// GEAR controller on top of VinePPO.
//
// This preserves VinePPO's episode/advantage logic through
// `gear_vineppo_episode_generator`, while replacing the trajectory inference
// strategy with the shared GEAR pruning/allocation controller.
{
  episode_generator+: {
    type: 'gear_vineppo_episode_generator',

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
      gear_pilot_branch_factor: $.gear.pilot_branch_factor,
      gear_likelihood_samples_per_distribution: $.gear.likelihood_samples_per_distribution,
      gear_tv_subnode_max_tokens: $.gear.tv_subnode_max_tokens,
      gear_tv_second_phase_tokens: $.gear.tv_second_phase_tokens,
      gear_tv_includes_half_factor: $.gear.tv_includes_half_factor,
      gear_n_min: $.gear.n_min,
      gear_budget_overhead_mode: $.gear.budget_overhead_mode,
      gear_allocation_mode: $.gear.allocation_mode,
      gear_use_residual_budget: $.gear.use_residual_budget,
      gear_budget_queue_count: $.gear.budget_queue_count,
      gear_budget_queue_capacity: $.gear.budget_queue_capacity,
      gear_budget_queue_timeout_seconds: $.gear.budget_queue_timeout_seconds,
      gear_skip_near_leaf_expand: $.gear.skip_near_leaf_expand,
      gear_root_allocation: $.gear.root_allocation,
      gear_eps_tail: $.gear.eps_tail,
      gear_eps_tail_calibration_path: $.gear.eps_tail_calibration_path,
      gear_eps_tail_by_depth: $.gear.eps_tail_by_depth,
      gear_bound_form: $.gear.bound_form,
      gear_tv_estimator: $.gear.tv_estimator,
      gear_strict_vdra: $.gear.strict_vdra,
      gear_invalid_support_policy: $.gear.invalid_support_policy,
      gear_budget_mode: $.gear.budget_mode,
      gear_allocation_proxy: $.gear.allocation_proxy,
    },

    gear_full_tree_demo_every_n_trees: $.gear.full_tree_demo_every_n_trees,
    gear_full_tree_demo_max_trees: $.gear.full_tree_demo_max_trees,
    gear_tree_policy_algorithm_name: $.gear.tree_policy_algorithm_name,
    gear_tree_policy_segmentation_type: $.gear.tree_policy_segmentation_type,
    gear_tree_policy_tree_shape: $.gear.tree_policy_tree_shape,
    gear_tree_policy_tree_m: $.gear.tree_policy_tree_m,
    gear_demos_dir: $.gear.demos_dir,
    gear_print_run_manifest: $.gear.print_run_manifest,
  },

  trainer+: {
    type: 'ppo',
    params+: {
      use_prob_mask: false,
    },
  },
}
