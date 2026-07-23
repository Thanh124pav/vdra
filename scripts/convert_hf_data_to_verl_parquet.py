#!/usr/bin/env python3
"""Download/cache math datasets and convert them to VERL-ready parquet.

The script supports two workflows:
  1. --source hf: pull public datasets from HuggingFace, save them with
     save_to_disk(), then convert to parquet.
  2. --source disk: convert already-saved HuggingFace DatasetDict directories.

Output parquet files deliberately use one fixed schema so multiple validation
files can be passed together via Hydra data.val_files=[...].
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import datasets


GSM8K_NATIVE_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
ANSWER_PREFIX_INSTRUCTION = "Let's think step by step. Put the final answer after a line exactly '# Answer'."

HF_SOURCES: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "repo_id": "gsm8k",
        "config_name": "main",
        "revision": "e53f048856ff4f594e959d75785d2c2d37b678ee",
    },
    "math": {
        "repo_id": "EleutherAI/hendrycks_math",
        "config_names": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        "test_repo_id": "HuggingFaceH4/MATH-500",
        "test_source_config": "MATH-500",
    },
    "math500": {
        "repo_id": "HuggingFaceH4/MATH-500",
    },
    "aime24": {
        "repo_id": "math-ai/aime24",
        "revision": "83a7f387baaa524a8bda0022eac0541582297103",
    },
    "aime25": {
        "repo_id": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    },
    "amc23": {
        "repo_id": "math-ai/amc23",
        "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
    },
    "olympiadbench_hf": {
        "repo_id": "Hothan/OlympiadBench",
        "config_name": "OE_TO_maths_en_COMP",
        "revision": "91184b52131e7fc9455fef848035173aea8cc01a",
    },
}

# Kept as fixed nullable-string fields so pyarrow can concatenate any mix of
# output parquet files. Do not put raw ints/bools/lists here: they create
# incompatible nested struct schemas across datasets.
EXTRA_STRING_FIELDS = (
    "dataset",
    "split",
    "source_repo",
    "source_config",
    "id",
    "unique_id",
    "question_number",
    "subject",
    "subfield",
    "level",
    "difficulty",
    "url",
    "license",
    "data_topic",
    "answer_type",
    "type",
    "is_multiple_answer",
    "unit",
    "error",
    "question_type",
    "language",
    "modality",
    "question",
    "problem",
    "answer",
    "solution",
    "final_answer",
    "_provided_sol",
)

ROW_FEATURES = datasets.Features(
    {
        "data_source": datasets.Value("string"),
        "prompt": [
            {
                "role": datasets.Value("string"),
                "content": datasets.Value("string"),
            }
        ],
        "ability": datasets.Value("string"),
        "reward_model": {
            "style": datasets.Value("string"),
            "ground_truth": datasets.Value("string"),
        },
        "extra_info": {
            "index": datasets.Value("int64"),
            **{field: datasets.Value("string") for field in EXTRA_STRING_FIELDS},
        },
    }
)


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Adapter:
    name: str
    question_field: str
    answer_fn: Callable[[dict[str, Any]], str]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return "\n\n".join(_clean_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _strip_commas(value: str) -> str:
    return value.replace(",", "").strip()


def _extract_gsm8k_answer(example: dict[str, Any]) -> str:
    raw = _clean_text(example.get("answer"))
    match = re.search(r"####\s*(.+?)\s*$", raw, flags=re.DOTALL)
    if not match:
        raise ConversionError("GSM8K answer is missing a final '#### ...' marker")
    return _strip_commas(match.group(1))


def _last_boxed(text: str) -> str | None:
    marker = "\\boxed"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    rest = text[idx + len(marker) :].lstrip()
    if rest.startswith("{"):
        depth = 0
        chars: list[str] = []
        for char in rest:
            if char == "{":
                if depth > 0:
                    chars.append(char)
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
                chars.append(char)
            else:
                if depth > 0:
                    chars.append(char)
        return None
    if rest:
        return rest.split("$", 1)[0].strip()
    return None


def _zero_answer_fallback(example: dict[str, Any]) -> str | None:
    """Handle a tiny known-bad MATH pattern: empty boxed answer but proof says zero."""

    solution = _clean_text(example.get("solution")).lower()
    if "\\boxed{}" not in solution:
        return None
    zero_patterns = (
        "there are $\\boxed{}$ primes",
        "not prime for any",
        "no prime",
        "no primes",
        "0 primes",
    )
    if any(pattern in solution for pattern in zero_patterns):
        return "0"
    return None


def _answer_from_field(*fields: str, boxed_fallback_fields: Iterable[str] = ()) -> Callable[[dict[str, Any]], str]:
    def answer_fn(example: dict[str, Any]) -> str:
        for field in fields:
            answer = _clean_text(example.get(field))
            if answer:
                return answer
        for field in boxed_fallback_fields:
            boxed = _last_boxed(_clean_text(example.get(field)))
            if boxed:
                return boxed
        zero = _zero_answer_fallback(example)
        if zero is not None:
            return zero
        raise ConversionError(f"missing answer in fields={fields} boxed_fallback_fields={tuple(boxed_fallback_fields)}")

    return answer_fn


def _boxed_from_field(field: str, fallback_fields: Iterable[str] = ()) -> Callable[[dict[str, Any]], str]:
    def answer_fn(example: dict[str, Any]) -> str:
        boxed = _last_boxed(_clean_text(example.get(field)))
        if boxed:
            return boxed
        for fallback in fallback_fields:
            answer = _clean_text(example.get(fallback))
            if answer:
                return answer
        raise ConversionError(f"missing boxed answer in field={field!r}")

    return answer_fn


def _olympiadbench_answer(example: dict[str, Any]) -> str:
    for field in ("answer", "final_answer"):
        answer = _clean_text(example.get(field))
        if answer:
            return answer
    raise ConversionError("missing olympiadbench answer/final_answer")


def _olympiadbench_hf_answer(example: dict[str, Any]) -> str:
    answer = _clean_text(example.get("final_answer"))
    if not answer:
        raise ConversionError("missing olympiadbench_hf final_answer")
    return answer


ADAPTERS: dict[str, Adapter] = {
    "gsm8k": Adapter("gsm8k", "question", _extract_gsm8k_answer),
    "math": Adapter("math", "problem", _answer_from_field("answer", boxed_fallback_fields=("solution",))),
    "math500": Adapter("math500", "problem", _answer_from_field("answer", boxed_fallback_fields=("solution",))),
    "aime24": Adapter("aime24", "problem", _boxed_from_field("solution")),
    "aime25": Adapter("aime25", "problem", _answer_from_field("answer")),
    "amc23": Adapter("amc23", "question", _answer_from_field("answer")),
    "collegeMath": Adapter("collegeMath", "problem", _answer_from_field("answer")),
    "olympiadbench": Adapter("olympiadbench", "problem", _olympiadbench_answer),
    "olympiadbench_hf": Adapter("olympiadbench_hf", "question", _olympiadbench_hf_answer),
}


def adapter_for_dataset(name: str) -> Adapter:
    if name.startswith("math-local-"):
        return Adapter("math-local", "problem", _answer_from_field("answer", boxed_fallback_fields=("solution",)))
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise ConversionError(f"no adapter for dataset {name!r}") from exc


def _is_excluded(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _instruction_for(dataset_name: str, prompt_style: str) -> str:
    if prompt_style == "boxed":
        return BOXED_INSTRUCTION
    if prompt_style == "answer-prefix":
        return ANSWER_PREFIX_INSTRUCTION
    if prompt_style == "native":
        return GSM8K_NATIVE_INSTRUCTION if dataset_name == "gsm8k" else BOXED_INSTRUCTION
    raise ConversionError(f"unknown prompt style: {prompt_style}")


def _datasetdict_from_loaded(loaded: datasets.Dataset | datasets.DatasetDict) -> datasets.DatasetDict:
    if isinstance(loaded, datasets.DatasetDict):
        return loaded
    if isinstance(loaded, datasets.Dataset):
        return datasets.DatasetDict({"train": loaded})
    raise ConversionError(f"expected Dataset or DatasetDict, got {type(loaded)!r}")


def _with_source_config(dataset: datasets.DatasetDict, config_name: str) -> datasets.DatasetDict:
    return datasets.DatasetDict(
        {
            split: split_dataset.map(lambda _: {"__source_config": config_name})
            for split, split_dataset in dataset.items()
        }
    )


def _load_hf_dataset(dataset_name: str) -> datasets.DatasetDict:
    if dataset_name not in HF_SOURCES:
        raise ConversionError(f"dataset {dataset_name!r} has no HuggingFace source mapping")
    source = HF_SOURCES[dataset_name]
    kwargs = {"revision": source["revision"]} if source.get("revision") else {}
    repo_id = source["repo_id"]
    config_names = source.get("config_names")
    if config_names:
        per_config = [
            _with_source_config(_datasetdict_from_loaded(datasets.load_dataset(repo_id, config, **kwargs)), config)
            for config in config_names
        ]
        splits = sorted({split for dataset in per_config for split in dataset.keys()})
        merged = datasets.DatasetDict()
        for split in splits:
            parts = [dataset[split] for dataset in per_config if split in dataset]
            merged[split] = datasets.concatenate_datasets(parts)
        test_repo_id = source.get("test_repo_id")
        if test_repo_id:
            test_loaded = _datasetdict_from_loaded(datasets.load_dataset(test_repo_id))
            if "test" not in test_loaded:
                raise ConversionError(f"{test_repo_id} does not provide a test split")
            merged["test"] = test_loaded["test"].map(
                lambda _: {"__source_config": str(source.get("test_source_config", test_repo_id))}
            )
        return merged
    config_name = source.get("config_name")
    if config_name:
        loaded = datasets.load_dataset(repo_id, config_name, **kwargs)
        return _with_source_config(_datasetdict_from_loaded(loaded), str(config_name))
    loaded = datasets.load_dataset(repo_id, **kwargs)
    return _datasetdict_from_loaded(loaded)


def _load_or_download_dataset(
    dataset_name: str,
    *,
    input_root: Path,
    source: str,
    overwrite_raw: bool,
) -> tuple[datasets.DatasetDict, str, str]:
    disk_path = input_root / dataset_name
    if source in {"auto", "disk"} and (disk_path / "dataset_dict.json").exists():
        return _datasetdict_from_loaded(datasets.load_from_disk(str(disk_path))), "disk", str(disk_path)
    if source == "disk":
        raise ConversionError(f"missing saved dataset: {disk_path}")

    dataset = _load_hf_dataset(dataset_name)
    if overwrite_raw and disk_path.exists():
        shutil.rmtree(disk_path)
    if not disk_path.exists():
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(disk_path))
    return dataset, "hf", HF_SOURCES[dataset_name]["repo_id"]


def _check_text_only_olympiadbench_hf(dataset: datasets.DatasetDict) -> None:
    for split, split_dataset in dataset.items():
        if "modality" in split_dataset.column_names:
            modalities = {str(value) for value in split_dataset["modality"] if value is not None}
            if modalities - {"Text-only"}:
                raise ConversionError(f"olympiadbench_hf/{split} has non-text modalities: {sorted(modalities)}")
        image_columns = [col for col in split_dataset.column_names if col.startswith("image_")]
        for col in image_columns:
            if any(value is not None for value in split_dataset[col]):
                raise ConversionError(f"olympiadbench_hf/{split} has non-null image column {col}")


def _source_repo(dataset_name: str) -> str:
    return str(HF_SOURCES.get(dataset_name, {}).get("repo_id", ""))


def _source_config(dataset_name: str) -> str:
    source = HF_SOURCES.get(dataset_name, {})
    if "config_name" in source:
        return str(source.get("config_name", ""))
    if "config_names" in source:
        configs = ",".join(str(item) for item in source.get("config_names", ()))
        if source.get("test_source_config"):
            return f"train:{configs};test:{source['test_source_config']}"
        return configs
    return ""


def _build_extra_info(dataset_name: str, split: str, idx: int, example: dict[str, Any]) -> dict[str, Any]:
    extra = {field: "" for field in EXTRA_STRING_FIELDS}
    extra["dataset"] = str(dataset_name)
    extra["split"] = str(split)
    extra["source_repo"] = _source_repo(dataset_name)
    extra["source_config"] = _clean_text(example.get("__source_config")) or _source_config(dataset_name)
    for field in EXTRA_STRING_FIELDS:
        if field in {"dataset", "split", "source_repo", "source_config"}:
            continue
        if field in example and example[field] is not None:
            extra[field] = _clean_text(example[field])
    if not extra.get("subject") and example.get("type") is not None:
        extra["subject"] = _clean_text(example.get("type"))
    if not extra.get("question_type") and example.get("type") is not None:
        extra["question_type"] = _clean_text(example.get("type"))
    extra["index"] = int(idx)
    return extra


def _convert_example(
    dataset_name: str,
    split: str,
    idx: int,
    example: dict[str, Any],
    adapter: Adapter,
    prompt_style: str,
) -> dict[str, Any]:
    question = _clean_text(example.get(adapter.question_field))
    if not question:
        raise ConversionError(f"missing question field {adapter.question_field!r}")
    ground_truth = _clean_text(adapter.answer_fn(example))
    if not ground_truth:
        raise ConversionError("empty ground_truth")
    content = f"{question} {_instruction_for(dataset_name, prompt_style)}".strip()
    return {
        "data_source": dataset_name,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": _build_extra_info(dataset_name, split, idx, example),
    }


def _verify_parquet(path: Path, expected_rows: int) -> None:
    loaded = datasets.load_dataset("parquet", data_files=str(path))["train"]
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required - set(loaded.column_names)
    if missing:
        raise ConversionError(f"{path} missing required columns: {sorted(missing)}")
    if len(loaded) != expected_rows:
        raise ConversionError(f"{path} row count mismatch: {len(loaded)} != {expected_rows}")
    for idx, row in enumerate(loaded):
        prompt = row["prompt"]
        reward_model = row["reward_model"]
        if not prompt or not isinstance(prompt, list) or not prompt[0].get("content"):
            raise ConversionError(f"{path} row {idx} has invalid prompt")
        if not reward_model or not _clean_text(reward_model.get("ground_truth")):
            raise ConversionError(f"{path} row {idx} has empty reward_model.ground_truth")


def convert_dataset(
    dataset_name: str,
    dataset: datasets.DatasetDict,
    output_root: Path,
    *,
    prompt_style: str,
    source_kind: str,
    source_ref: str,
) -> list[dict[str, Any]]:
    adapter = adapter_for_dataset(dataset_name)
    if dataset_name == "olympiadbench_hf":
        _check_text_only_olympiadbench_hf(dataset)

    rows: list[dict[str, Any]] = []
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_dataset in dataset.items():
        converted = []
        for idx, example in enumerate(split_dataset):
            try:
                converted.append(_convert_example(dataset_name, split, idx, dict(example), adapter, prompt_style))
            except ConversionError as exc:
                raise ConversionError(f"{dataset_name}/{split}[{idx}]: {exc}") from exc
        output_path = output_dir / f"{split}.parquet"
        datasets.Dataset.from_list(converted, features=ROW_FEATURES).to_parquet(str(output_path))
        _verify_parquet(output_path, expected_rows=len(split_dataset))
        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "source_kind": source_kind,
                "source_ref": source_ref,
                "output_path": str(output_path),
                "row_count": len(split_dataset),
                "adapter": adapter.name,
                "prompt_style": prompt_style,
            }
        )
    return rows


def _resolve_dataset_names(input_root: Path, source: str, requested: list[str], excludes: list[str]) -> list[str]:
    if requested and requested != ["all"]:
        names = requested
    elif source == "hf":
        names = sorted(HF_SOURCES)
    else:
        disk_names = sorted(path.name for path in input_root.iterdir() if (path / "dataset_dict.json").exists()) if input_root.exists() else []
        names = sorted(set(disk_names) | (set(HF_SOURCES) if source == "auto" else set()))
    return [name for name in names if not _is_excluded(name, excludes)]


def _verify_concat(paths: list[str]) -> None:
    if len(paths) <= 1:
        return
    loaded = datasets.load_dataset("parquet", data_files=paths)["train"]
    print(f"[verify-concat] loaded {len(paths)} files, rows={len(loaded)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "disk", "hf"], default="auto")
    parser.add_argument("--input-root", default="data", type=Path, help="Raw save_to_disk root.")
    parser.add_argument("--output-root", default="verl/data", type=Path, help="VERL parquet output root.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset names or 'all'.")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--schema", default="verl-ready", choices=["verl-ready"])
    parser.add_argument("--prompt-style", choices=["boxed", "answer-prefix", "native"], default="boxed")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--verify-concat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    dataset_names = _resolve_dataset_names(input_root, args.source, args.datasets, args.exclude)
    for dataset_name in dataset_names:
        print(f"[prepare] {dataset_name}")
        dataset, source_kind, source_ref = _load_or_download_dataset(
            dataset_name,
            input_root=input_root,
            source=args.source,
            overwrite_raw=args.overwrite_raw,
        )
        print(f"[convert] {dataset_name} source={source_kind}:{source_ref}")
        manifest.extend(
            convert_dataset(
                dataset_name,
                dataset,
                output_root,
                prompt_style=args.prompt_style,
                source_kind=source_kind,
                source_ref=source_ref,
            )
        )

    manifest_path = output_root / "_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.verify_concat:
        _verify_concat([item["output_path"] for item in manifest])
    print(f"[done] wrote {len(manifest)} split files")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
