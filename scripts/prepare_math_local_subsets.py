#!/usr/bin/env python3
"""Create tiny HuggingFace DatasetDict slices from the downloaded MATH data.

The training configs load `DatasetDict.load_from_disk`, so these subsets keep
the same on-disk format as `data/math` while shrinking train/validation/test to
small deterministic slices for local end-to-end checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict


def select_n(ds, n: int):
    n = min(int(n), len(ds))
    return ds.select(range(n))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/math")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 30, 100])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    output_root = Path(args.output_root)
    dataset = DatasetDict.load_from_disk(str(source))

    for size in args.sizes:
        out = output_root / f"math-local-{size}"
        subset = DatasetDict()
        for split, split_ds in dataset.items():
            shuffled = split_ds.shuffle(seed=args.seed)
            subset[split] = select_n(shuffled, size)
        subset.save_to_disk(str(out))
        print(f"{out}: " + ", ".join(f"{k}={len(v)}" for k, v in subset.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
