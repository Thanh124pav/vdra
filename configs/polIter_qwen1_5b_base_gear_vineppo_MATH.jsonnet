// GEAR-VinePPO on MATH with Qwen2.5-1.5B-base.
//
// Starts from the SPO-chain base, switches to VinePPO, then applies GEAR to
// the trajectory generation strategy.  Keep TREE=6 unless intentionally
// testing deeper trees, because VinePPO expects root->response trajectories.
(import 'polIter_qwen1_5b_base_spo_chain_MATH.jsonnet')
+ (import 'algorithms/vineppo.libsonnet')
+ (import 'gear_defaults.libsonnet')
+ {
  gear+: {
    tree_policy_algorithm_name: 'gear_vineppo',
    tree_policy_segmentation_type: 'vineppo_step',
    skip_near_leaf_expand: false,
  },
}
+ (import 'gear_vineppo_overlay.libsonnet')
