// GEAR-SPO-chain on MATH with Qwen2.5-1.5B-base.
//
// Starts from SPO-chain and applies the shared GEAR controller.  The launcher
// defaults TREE=6 so the run remains chain-compatible while still enabling
// root-level k/prune/allocation diagnostics.
(import 'polIter_qwen1_5b_base_spo_chain_MATH.jsonnet')
+ (import 'gear_defaults.libsonnet')
+ {
  gear+: {
    tree_policy_algorithm_name: 'gear_spo_chain',
    tree_policy_segmentation_type: 'spo_chain_step',
    skip_near_leaf_expand: false,
  },
}
+ (import 'gear_overlay.libsonnet')
