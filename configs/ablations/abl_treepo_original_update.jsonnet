// TreePO-style original update objective ablation.
{
  gear+: {
    tree_update_mode: 'treepo_original',
    treepo_global_weight: 0.5,
  },
  episode_generator+: {
    tree_update_mode: 'treepo_original',
    treepo_global_weight: 0.5,
  },
}
