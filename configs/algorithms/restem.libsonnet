// RestEM — Rejection sampling + EM-style fine-tuning.
//
// Sample many rollouts, keep only those that solve the task, fine-tune the
// policy on them, repeat.  Equivalent to STaR / Self-Taught Reasoner.
{
  episode_generator+: {
    type: 'math_restem_episode_generator',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: { type: 'restem' },
}
