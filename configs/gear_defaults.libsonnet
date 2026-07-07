// Default GEAR hyperparameters.
{
  gear: {


    epsilon: 0.02,
    r_max: 1.0,
    gamma: 0.9,

    // Tail-fill concurrency to vLLM /completions
    score_concurrency: 64,
    score_timeout_seconds: 120.0,
    score_retry_attempts: 5,
    score_retry_backoff_seconds: 0.5,

    // Online GEAR: predict k per node, then allocate branch budget across queues.
    k_algorithm: 'hierarchical',
    generation_mode: 'single_request',
    n_tv_estimates: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
    tv_includes_half_factor: true,
    budget_lambda: 0.02,
    n_min: 0,
    budget_overhead_mode: 'flexible',
    allocation_mode: 'budget_allocation',
    use_residual_budget: true,
    budget_queue_count: 4,
    budget_queue_timeout_seconds: 1.0,
    // When true, skip TV/budget allocation at the final expansion depth and
    // use uniform SPO-style branch factor B instead. This avoids near-leaf
    // context exhaustion from TV probe continuations.
    skip_near_leaf_expand: true,
    // When true, estimate root reward variance for every problem in the
    // current minibatch and allocate the depth-0 branch budget across those
    // roots before constructing each individual tree.
    root_allocation: true,

    // Edge handling
    // Update objective for tree-policy training. `spo` preserves the
    // existing local parent-child advantage. Use `treepo_original` or
    // `treerl_original` for reproduction ablations.
    tree_update_mode: 'spo',
    treepo_global_weight: 0.5,
    treerl_gamma: 0.9,
    zero_advantage_when_pruned: true,
    emit_pruned_edges: false,

    // Logging: how many SHARE / PRUNE demo rows to dump per tree.
    // Set 0 to skip demo files entirely (per-depth rates stay on).
    demo_examples_per_tree: 4,

    // Where to write `demos.jsonl` + `demos.md` (offline-friendly). When
    // null they go under <APP_DIRECTORY>/<exp>/gear_demos/. Override with
    // an absolute path if you want them somewhere shared.
    demos_dir: null,

    // Off by default so an offline server with `wandb mode=offline` does
    // not try to upload tables. Flip to true if you do have wandb running.
    log_demos_to_wandb: false,

    // Full-tree demos are rate-limited, but when emitted they keep full text.
    full_tree_demo_every_n_trees: 0,
    full_tree_demo_max_trees: 5,
    tree_policy_algorithm_name: 'gear_spo',
    tree_policy_segmentation_type: 'spo_step',
    tree_policy_tree_shape: null,
    tree_policy_tree_m: null,
    print_run_manifest: true,
  },
}
