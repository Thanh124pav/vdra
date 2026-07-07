// GEAR-SPO-chain on MATH with DeepSeek-R1-Distill-Qwen-1.5B.
(import 'polIter_deepseekR1Qwen_spo_chain_MATH.jsonnet')
+ (import 'gear_defaults.libsonnet')
+ {
  gear+: {
    tree_policy_algorithm_name: 'gear_spo_chain',
    tree_policy_segmentation_type: 'spo_chain_step',
    skip_near_leaf_expand: false,
  },
}
+ (import 'gear_overlay.libsonnet')
