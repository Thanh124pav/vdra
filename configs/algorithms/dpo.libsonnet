// DPO (positive variant) — Direct Preference Optimization.
//
// Offline pairwise contrastive training.  The episode generator mines (good,
// bad) trajectory pairs from a buffer; the trainer fits the DPO loss without
// an explicit reward model.
{
  episode_generator+: {
    type: 'math_dpo_positive_episode_generator',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: { type: 'dpo_positive' },
}
