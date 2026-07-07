import asyncio
import json
import time
from typing import List, Optional, Dict, Any

from datasets import Dataset

import guidance
from guidance.llms import OpenAI, OpenAIVLLM
from treetune import logging_utils
from treetune.common import guidance_utils as gu, Registrable, Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.tree_inference import Node
from treetune.inference_strategies.tree_inference.answer_extraction import (
    AnswerExtractor,
)
from treetune.inference_strategies.tree_inference.expansion import NodeExpander
from treetune.tokenization_utils.base_tokenizer import Tokenizer

logger = logging_utils.get_logger(__name__)

TREE_COLNAME = "_treetune__reasoning_tree"


class GuidanceLLM(Registrable):
    pass


class FilterFn(Registrable):
    def __call__(self, example: Dict[str, Any]) -> bool:
        raise NotImplementedError


@GuidanceLLM.register("openai", exist_ok=True)
class OpenAIGuidanceLLM(OpenAI, GuidanceLLM):
    pass


@GuidanceLLM.register("openai_vllm", exist_ok=True)
class OpenAIVLLMGuidanceLLM(OpenAIVLLM, GuidanceLLM):
    pass


@FilterFn.register("keep_invalid_value", exist_ok=True)
class KeepInvalidValueFilterFn(FilterFn):
    def __init__(self, invalid_value: int, invalid_value_field: str):
        self.invalid_value = invalid_value
        self.invalid_value_field = invalid_value_field

    def __call__(self, example: Dict[str, Any]) -> bool:
        return example[self.invalid_value_field] == self.invalid_value


@FilterFn.register("keep_non_last_steps", exist_ok=True)
class KeepNonLastStepsFilterFn(FilterFn):
    def __call__(self, example: Dict[str, Any]) -> bool:
        if "is_last_step" in example:
            return not example["is_last_step"]
        elif "gt_value" in example:
            return example["gt_value"] == -100
        else:
            raise ValueError("Invalid example format")


@InferenceStrategy.register("tree", exist_ok=True)
class TreeInferenceStrategy(InferenceStrategy):
    def __init__(
        self,
        max_depth: int,
        question_template: str,
        node_expander: NodeExpander,
        answer_extractor: AnswerExtractor,
        guidance_llm: Lazy[GuidanceLLM],
        question_field: str = "question",
        max_concurrent_programs: int = 128,
        max_concurrent_generations: int = 2048,
        seed: Optional[int] = None,
        max_question_length: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        filter_functions: Optional[List[FilterFn]] = None,
        no_cache: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.question_template = question_template
        self.node_expander = node_expander
        self.answer_extractor = answer_extractor
        self.seed = seed

        self.max_concurrent_programs = max_concurrent_programs
        self.max_concurrent_generations = max_concurrent_generations

        self.guidance_llm_lazy = guidance_llm
        self.question_field = question_field
        self.no_cache = no_cache

        self.max_question_length = max_question_length
        self.tokenizer = tokenizer
        if max_question_length is not None:
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided if max_question_length is provided"

        self.filter_functions = filter_functions or []

        self.node_expander.set_seed(seed)
        self.answer_extractor.set_seed(seed)
        if hasattr(self.node_expander, "set_answer_extractor"):
            self.node_expander.set_answer_extractor(self.answer_extractor)

        if self.log_level is not None:
            logger.setLevel(self.log_level)

    def generate(self, dataset: Dataset) -> Dataset:
        """
        Generate a new dataset based on the given dataset
        Params:
            dataset: The dataset to generate from

        Returns:
            A new dataset, which is the input dataset with new columns.
            New columns are prefixed with "_treetune__{column_name}".
            The following columns should be always added:
            - _treetune__candidate_answers: List[str] - The candidate answers
        """
        return asyncio.run(self._concurrent_generate(dataset))

    def get_temp_tree_dir(self):
        temp_tree_dir = self.result_dir / "trees"
        temp_tree_dir.mkdir(parents=True, exist_ok=True)
        return temp_tree_dir

    def get_tree_instance_path(self, instance_idx):
        return self.get_temp_tree_dir() / f"{instance_idx}.json"

    async def _concurrent_generate(self, dataset: Dataset) -> Dataset:
        # Create a semaphore to limit the number of concurrent programs
        sem_program = asyncio.Semaphore(self.max_concurrent_programs)

        # Create a semaphore to limit the number of concurrent generations
        sem_generation = asyncio.Semaphore(self.max_concurrent_generations)

        async def sem_run_program(*args, **kwargs):
            async with sem_program:
                return await gu.run_program(*args, **kwargs)

        async def wrapper_construct_tree(tree_idx, *args, **kwargs):
            async with sem_generation:
                try:
                    tr = await self._construct_tree(*args, **kwargs)
                    return tree_idx, tr
                except:
                    # If there is an error, we just exit the program
                    # as soon as possible, otherwise the program will continue
                    # blocking the semaphore and thus blocking the entire process
                    exit(1)

        # Set the guidance LLM
        guidance.llm = self.guidance_llm_lazy.construct()

        self.node_expander.set_run_program(sem_run_program)
        self.answer_extractor.set_run_program(sem_run_program)

        question_format_keys = []
        for column in dataset.column_names:
            if f"{{{column}}}" in self.question_template:
                question_format_keys.append(column)
        logger.info(f"Question format keys: {question_format_keys}")
        assert self.question_field in question_format_keys, (
            f"Question field '{self.question_field}' must be in the question template. "
            f"Available format keys: {question_format_keys}"
        )

        if self.max_question_length is not None:
            dataset = self._filter_out_long_questions(dataset, question_format_keys)

        before_filter_len = len(dataset)
        for filter_fn in self.filter_functions:
            dataset = dataset.filter(
                filter_fn,
                num_proc=4,
                desc=f"Applying filter function {filter_fn.__class__.__name__}",
            )
        logger.info(
            f"Filtered out {before_filter_len - len(dataset)} examples from {before_filter_len} examples."
        )

        tasks = []
        trees = {}
        from tqdm import tqdm as tqdm_iter

        for data_instance in tqdm_iter(
            dataset,
            desc="Creating concurrent asyncio tasks for tree construction...",
        ):
            instance_idx = data_instance["_treetune__idx"]

            if not self.no_cache:
                tree_file_path = self.get_tree_instance_path(instance_idx)
                try:
                    with tree_file_path.open("r") as f:
                        tree = json.load(f)
                        logger.info(f"Loaded tree from {tree_file_path}")
                    assert len(tree) > 0
                    trees[instance_idx] = tree
                    # Skip if the tree is already constructed
                    continue
                except FileNotFoundError:
                    pass
                except Exception as e:
                    # If the file exists but is corrupted, we log the error and re-construct the tree
                    logger.error(f"Error loading tree from {tree_file_path}: {e}")

            format_kwargs = {key: data_instance[key] for key in question_format_keys}
            initial_prompt = self.question_template.format(**format_kwargs)

            tasks.append(
                asyncio.create_task(
                    wrapper_construct_tree(
                        instance_idx,
                        initial_prompt,
                        self.max_depth,
                        data_instance=data_instance,
                    )
                )
            )

        # Report the current progress to cloud logger
        if self.cloud_logger is not None:
            self.cloud_logger.log({"construction_progress": len(trees) / len(dataset)})

        # Create a progress bar for the tree construction tasks
        from tqdm.asyncio import tqdm as tqdm_asyncio

        # Maintain a progress bar for the tree construction tasks.
        # It updates whenever any of the tasks finishes.
        for task in tqdm_asyncio.as_completed(tasks, desc="Constructing trees"):
            instance_idx, tree = await task
            trees[instance_idx] = tree

            if not self.no_cache:
                tree_file_path = self.get_tree_instance_path(instance_idx)
                with tree_file_path.open("w") as f:  # so we can resume later on
                    json.dump(tree, f)

            if self.cloud_logger is not None:
                self.cloud_logger.log(
                    {"construction_progress": len(trees) / len(dataset)}
                )

        trees = [
            trees[idx] for idx in dataset["_treetune__idx"]
        ]  # change order back to original
        assert len(trees) == len(
            dataset
        ), f"len(trees)={len(trees)}, len(dataset)={len(dataset)}"

        # Utility function to create a dataset
        def create_column(column_name, extraction_method):
            return Dataset.from_dict(
                {column_name: [extraction_method(tree) for tree in trees]}
            )[column_name]

        # Add new columns to the dataset
        dataset = dataset.add_column(
            TREE_COLNAME, create_column("tree", self._convert_tree_to_string)
        )
        dataset = dataset.add_column(
            "_treetune__candidate_answers",
            create_column("answer", self._extract_answer_candidates_from_tree),
        )
        dataset = dataset.add_column(
            "_treetune__candidate_logprobs",
            create_column("logprobs", self._extract_candidates_logprobs_from_tree),
        )
        dataset = dataset.add_column(
            "_treetune__candidate_num_tokens",
            create_column("num_tokens", self._extract_candidates_num_tokens_from_tree),
        )

        return dataset

    def _filter_out_long_questions(self, dataset, question_format_keys):
        tokenizer = self.tokenizer
        max_question_length = self.max_question_length
        question_template = self.question_template

        def filter_long_questions(example):
            format_kwargs = {key: example[key] for key in question_format_keys}
            prompt = question_template.format(**format_kwargs)
            tokens = tokenizer(prompt).input_ids
            return len(tokens) <= max_question_length

        dataset_len_before = len(dataset)
        dataset = dataset.filter(
            filter_long_questions, num_proc=4, desc="Filtering long questions"
        )
        logger.info(
            f"Filtered out {dataset_len_before - len(dataset)} long questions from {dataset_len_before} questions."
        )
        return dataset

    async def _construct_tree(
        self,
        initial_prompt: str,
        max_depth: int,
        data_instance: Optional[Dict[str, Any]] = None,
    ):
        t0_tree = time.time()
        # First, we create the root node
        tree = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
        }

        queue = asyncio.Queue()
        await queue.put((tree, initial_prompt, 0))

        num_workers = max(1, self.max_concurrent_programs)
        worker_errors = []

        async def worker() -> None:
            while True:
                node, prefix, depth = await queue.get()
                try:
                    if depth >= max_depth:
                        continue

                    children = await self.node_expander.expand(node, prefix, depth)
                    node["children"] = children

                    answer_extraction_tasks = []
                    answer_extraction_children = []

                    for child in children:
                        if child.get("stop_text") is None:
                            answer_extraction_tasks.append(
                                self.answer_extractor.extract_from_node(child)
                            )
                            answer_extraction_children.append(child)
                        else:
                            await queue.put((child, child["full_text"], depth + 1))

                    if answer_extraction_tasks:
                        answers = await asyncio.gather(*answer_extraction_tasks)
                        for child, answer in zip(answer_extraction_children, answers):
                            child["answer"] = answer
                except Exception as exc:
                    worker_errors.append(exc)
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(num_workers)
        ]

        await queue.join()

        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        if worker_errors:
            raise worker_errors[0]

        # Remove the `_request_object` field from the tree
        tree.pop("_request_object", None)

        tree["tree_construction_seconds"] = time.time() - t0_tree
        return tree

    def _convert_tree_to_string(self, tree: Node) -> str:
        # @TODO: Perhaps remove full_text to reduce the size of the tree
        tree_str = json.dumps(tree, indent=4, sort_keys=True)
        return tree_str

    def _extract_answer_candidates_from_tree(self, tree: Node) -> List[str]:
        candidates = []

        def dfs(node: Node) -> None:
            if "answer" in node:
                candidates.append(node["answer"])
            for child in node.get("children", []):
                dfs(child)

        dfs(tree)

        return candidates

    def _extract_candidates_logprobs_from_tree(self, tree: Node) -> List[float]:
        logprobs = []

        def dfs(node: Node, parent_sum_logprobs: float) -> None:
            if "answer" in node and "sum_logprobs" in node:
                logprobs.append(parent_sum_logprobs + node["sum_logprobs"])
            for child in node.get("children", []):
                dfs(child, parent_sum_logprobs + node.get("sum_logprobs", 0.0))

        dfs(tree, 0.0)

        return logprobs

    def _extract_candidates_num_tokens_from_tree(self, tree: Node) -> List[float]:
        num_tokens = []

        def dfs(node: Node) -> None:
            if "answer" in node and "num_tokens" in node:
                num_tokens.append(node["num_tokens"])
            for child in node.get("children", []):
                dfs(child)

        dfs(tree)

        return num_tokens
