"""Flat (non-tree) entry point for GRPO / RLOO on verl.

GRPO and RLOO are group-relative estimators over N independent completions per
prompt — not trees — so they run through verl's native ``run_ppo`` with the exact
built-in ``grpo`` / ``rloo`` advantage estimators, but using this recipe's
treetune-faithful PPO loss (``treetune_ppo``) and MATH reward (``gear_math``).

Select the estimator on the CLI, e.g. ``algorithm.adv_estimator=grpo``.
"""

from __future__ import annotations

import hydra

from verl.trainer.main_ppo import run_ppo

# Register treetune_ppo (policy loss) and gear_math (reward manager).
import recipe.gear_tree.policy_loss  # noqa: F401
import recipe.gear_tree.reward  # noqa: F401


@hydra.main(config_path="config", config_name="flat_trainer", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
