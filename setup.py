from setuptools import find_packages, setup

setup(
    name="treetune",
    version="0.2.0",
    description=(
        "Unified RL framework for reasoning LLMs. Supports PPO, GRPO, DPO, "
        "RestEM, VinePPO, RLOO, SPO-chain, SPO-tree, and GEAR."
    ),
    packages=find_packages(include=["treetune", "treetune.*", "guidance", "guidance.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "sortedcontainers",
        "httpx",
        "openai>=1.0",
    ],
)
