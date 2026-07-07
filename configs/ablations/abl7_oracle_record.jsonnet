// Abl 7: keep PRUNE/SHARE edges in the dataset so the analyzer can later
// audit them against the unmodified SPO baseline.
{ gear+: { emit_pruned_edges: true } }
+ {
  episode_generator+: {
    gear_emit_pruned_edges: true,
  },
}
