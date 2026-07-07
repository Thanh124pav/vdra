// Evaluation-only alias for plumbing tests on 4 GB GPUs.
(import 'polIter_qwen1_5b_base_gear_tree_MATH.jsonnet')
+ (import 'model_overrides/smollm_135m.jsonnet')
+ (import 'smollm_135m_for_MATH_eval.jsonnet')
