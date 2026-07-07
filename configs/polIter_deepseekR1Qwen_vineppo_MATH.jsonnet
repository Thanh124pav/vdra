// VinePPO on MATH with DeepSeek-R1-Distill-Qwen-1.5B.
// Reuse the model, task, long-CoT sampling, and MC rollout settings from the
// corresponding SPO-chain experiment, then switch to the VinePPO generator.
(import 'polIter_deepseekR1Qwen_spo_chain_MATH.jsonnet')
+ (import 'algorithms/vineppo.libsonnet')
