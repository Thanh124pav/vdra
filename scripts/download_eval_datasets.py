#!/usr/bin/env python3
"""Download the datasets used by the MATH evaluation pipelines."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import DatasetDict, load_dataset


DATASETS: Dict[str, Dict[str, Any]] = {
    "aime24": {
        "repo_id": "math-ai/aime24",
        "revision": "83a7f387baaa524a8bda0022eac0541582297103",
        "path": "aime24",
        "splits": {"test": 30},
        "required_columns": {"problem", "solution"},
    },
    "aime25": {
        "repo_id": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "path": "aime25",
        "splits": {"test": 30},
        "required_columns": {"problem", "answer"},
    },
    "amc23": {
        "repo_id": "math-ai/amc23",
        "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
        "path": "amc23",
        "splits": {"test": 40},
        "required_columns": {"question", "answer"},
    },
    "olympiadbench_hf": {
        "repo_id": "Hothan/OlympiadBench",
        "config_name": "OE_TO_maths_en_COMP",
        "revision": "91184b52131e7fc9455fef848035173aea8cc01a",
        "source_file": (
            "OlympiadBench/OE_TO_maths_en_COMP/"
            "OE_TO_maths_en_COMP.parquet"
        ),
        "path": "olympiadbench_hf",
        "splits": {"train": 674},
        "required_columns": {"question", "solution", "final_answer"},
    },
}


def validate_dataset(name: str, dataset: DatasetDict) -> None:
    spec = DATASETS[name]
    expected_splits = spec["splits"]
    if set(dataset) != set(expected_splits):
        raise ValueError(
            f"{name}: expected splits {sorted(expected_splits)}, "
            f"got {sorted(dataset)}"
        )

    required_columns = spec["required_columns"]
    for split, expected_rows in expected_splits.items():
        actual_rows = len(dataset[split])
        if actual_rows != expected_rows:
            raise ValueError(
                f"{name}/{split}: expected {expected_rows} rows, got {actual_rows}"
            )
        missing = required_columns - set(dataset[split].column_names)
        if missing:
            raise ValueError(
                f"{name}/{split}: missing required columns {sorted(missing)}"
            )


def download_dataset(name: str, data_dir: Path) -> Path:
    spec = DATASETS[name]
    destination = data_dir / spec["path"]
    if destination.exists():
        dataset = DatasetDict.load_from_disk(str(destination))
        validate_dataset(name, dataset)
        print(f"{name}: using existing {destination}")
        return destination

    load_args = [spec["repo_id"]]
    if spec.get("config_name") is not None:
        load_args.append(spec["config_name"])
    dataset = load_dataset(*load_args, revision=spec["revision"])
    validate_dataset(name, dataset)

    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    metadata = {
        key: spec[key]
        for key in ("repo_id", "config_name", "revision", "source_file")
        if key in spec
    }
    (destination / "_source.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{name}: saved to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        help="Datasets to download. Defaults to all eval datasets.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.datasets or sorted(DATASETS)
    for name in selected_datasets:
        download_dataset(name, args.data_dir)


if __name__ == "__main__":
    main()
