// GEAR-tree on MATH with Qwen2.5-1.5B-base.
// Inherits the SPO-tree config and applies the GEAR overlay.

(import 'polIter_qwen1_5b_base_spo_tree_MATH.jsonnet')
+ (import 'gear_defaults.libsonnet')
+ (import 'gear_overlay.libsonnet')
