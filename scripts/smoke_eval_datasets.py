#!/usr/bin/env python3
"""Run a tiny generation smoke test over every local MATH eval dataset."""

import argparse
import json
from pathlib import Path
import sys

import _jsonnet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from treetune.common import Params
from treetune.tasks import Task


EVAL_TASKS = (
    ("aime24", "aime24_inplace_no_answer_prefix.jsonnet", "test"),
    ("aime25", "aime25_inplace_no_answer_prefix.jsonnet", "test"),
    ("amc23", "amc23_inplace_no_answer_prefix.jsonnet", "test"),
    (
        "olympiadbench_hf",
        "olympiadbench_hf_inplace_no_answer_prefix.jsonnet",
        "train",
    ),
    ("collegeMath", "collegeMath_inplace_no_answer_prefix.jsonnet", "test"),
)


def load_prompts() -> list[tuple[str, str]]:
    prompts = []
    for dataset_name, config_name, split in EVAL_TASKS:
        config_path = ROOT / "configs" / "tasks" / config_name
        config = json.loads(_jsonnet.evaluate_file(str(config_path)))
        config["hf_num_proc"] = 1
        task = Task.from_params(Params(config))
        example = task.get_datasets(split)[0]
        prompts.append((dataset_name, example["query"]))
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Small causal LM used only to validate tokenization and generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading missing model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    for dataset_name, prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "prompt_chars": len(prompt),
                    "generated": generated,
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
