// TreeRL-style dense process/TD update objective ablation.
{
  gear+: {
    tree_update_mode: 'treerl_original',
    treerl_gamma: 0.9,
  },
  episode_generator+: {
    tree_update_mode: 'treerl_original',
    treerl_gamma: 0.9,
  },
}
