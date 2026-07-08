#!/usr/bin/env bash
# Create a dedicated CPU-capable environment for verl v0.6.0 unit tests /
# config compilation. vLLM + flash-attn are intentionally omitted (GPU-only;
# needed later for end-to-end rollout, not for importing core_algos or running
# the recipe's numeric parity tests).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

ENV=verl060
if ! conda env list | grep -q "/${ENV}$"; then
  conda create -y -n "$ENV" python=3.10
fi
conda activate "$ENV"

python -m pip install --upgrade pip
# torch CPU wheel (compatible with tensordict 0.8-0.10)
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
# Keep the CPU env focused on recipe imports and unit tests. Dataset/E2E helpers
# can be installed later with the GPU/vLLM runtime to avoid httpx resolver churn.
conda install -y -n "$ENV" numpy pytest requests certifi
# transformers 4.48.x still exposes AutoModelForVision2Seq (removed in >=4.49),
# which verl v0.6.0/verl/utils/model.py imports at module load.
python -m pip install \
  "transformers==4.48.3" \
  "tensordict==0.10.0" \
  "omegaconf" "hydra-core" "ray>=2.41.0" "codetiming" "dill" \
  "sympy" "pylatexenc" "regex" "pybind11"

echo "=== verify verl v0.6.0 + recipe import ==="
cd /home/pavt1024/vai/vdra
PYTHONPATH=verl python -c "
from verl.trainer.ppo.core_algos import get_policy_loss_fn
import recipe.gear_tree.policy_loss as pl
assert get_policy_loss_fn('treetune_ppo') is pl.compute_policy_loss_treetune
print('OK: verl v0.6.0 core_algos + treetune_ppo registered')
"
echo "ENV_SETUP_DONE"
