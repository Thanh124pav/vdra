// GEAR-VinePPO on MATH with DeepSeek-R1-Distill-Qwen-1.5B.
(import 'polIter_deepseekR1Qwen_vineppo_MATH.jsonnet')
+ (import 'gear_defaults.libsonnet')
+ {
  gear+: {
    tree_policy_algorithm_name: 'gear_vineppo',
    tree_policy_segmentation_type: 'vineppo_step',
    skip_near_leaf_expand: false,
  },
}
+ (import 'gear_vineppo_overlay.libsonnet')
