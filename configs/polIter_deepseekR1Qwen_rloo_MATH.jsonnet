// RLOO shares the DeepSeek-R1-Qwen MATH rollout and trainer settings with
// GRPO; only the group-advantage estimator changes to leave-one-out.
(import 'polIter_deepseekR1Qwen_grpo_MATH.jsonnet')
+ (import 'algorithms/rloo.libsonnet')
