import json
import os
from pathlib import Path
import subprocess

import _jsonnet
import pytest

from treetune.common import Params
from treetune.runtime.policy_iteration_runtime import print_evaluation_result
from treetune.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"


def test_evaluation_result_is_printed_to_terminal(capsys, tmp_path):
    result_dir = tmp_path / "inference_results"
    summary_path = tmp_path / "evaluation_results.jsonl"
    print_evaluation_result(
        "aime24_test",
        [{"once_hit": 0.25, "majority_vote": 0.2}],
        result_dir,
        summary_path,
        "checkpoint/hf_pretrained",
    )

    output = capsys.readouterr().out
    assert "=== Evaluation result: aime24_test ===" in output
    assert '"majority_vote": 0.2' in output
    assert '"once_hit": 0.25' in output
    assert f"Predictions: {result_dir}" in output
    assert f"Summary file: {summary_path}" in output

    record = json.loads(summary_path.read_text().strip())
    assert record == {
        "inference_name": "aime24_test",
        "metrics": [{"majority_vote": 0.2, "once_hit": 0.25}],
        "model": "checkpoint/hf_pretrained",
        "result_dir": str(result_dir),
    }


def test_math_eval_configs_include_shared_benchmarks():
    eval_configs = (
        "deepseekR1Qwen_for_MATH_eval.jsonnet",
        "deepseekR1Qwen7B_for_MATH_eval.jsonnet",
        "qwen1_5b_base_for_MATH_eval.jsonnet",
        "sft_deepseekmath_for_MATH_eval.jsonnet",
        "sft_rho1b_for_MATH_eval.jsonnet",
    )

    for config_name in eval_configs:
        config = (CONFIGS / config_name).read_text()
        assert "evaluation/math_benchmarks.libsonnet" in config
        assert "] + math_benchmark_pipelines" in config


def test_math_benchmark_overlay_defines_all_requested_evals():
    overlay = (CONFIGS / "evaluation" / "math_benchmarks.libsonnet").read_text()

    for inference_name in (
        "aime24_test",
        "aime25_test",
        "amc23_test",
        "olympiadbench_test",
        "collegeMath_test",
    ):
        assert f"inference_name: '{inference_name}'" in overlay


def test_downloaded_tasks_use_local_normalized_math_dataset_fields():
    expected = {
        "aime24_inplace_no_answer_prefix.jsonnet": (
            "'data/aime24'",
            "'problem'",
            "null",
            "'solution'",
        ),
        "aime25_inplace_no_answer_prefix.jsonnet": (
            "'data/aime25'",
            "'problem'",
            "'answer'",
            "null",
        ),
        "amc23_inplace_no_answer_prefix.jsonnet": (
            "'data/amc23'",
            "'question'",
            "'answer'",
            "null",
        ),
        "olympiadbench_hf_inplace_no_answer_prefix.jsonnet": (
            "'data/olympiadbench_hf'",
            "'question'",
            "'final_answer'",
            "'solution'",
        ),
    }

    for config_name, fields in expected.items():
        dataset_path, problem_field, answer_field, solution_field = fields
        config = (CONFIGS / "tasks" / config_name).read_text()
        assert f"dataset_dict_path: {dataset_path}" in config
        assert "load_dataset_dict: true" in config
        assert f"problem_field: {problem_field}" in config
        assert f"answer_field: {answer_field}" in config
        assert f"solution_field: {solution_field}" in config
        assert "normalize_dataset_fields: true" in config
        assert "use_dataset_answer: true" in config


@pytest.mark.parametrize(
    ("config_name", "dataset_path", "split"),
    (
        ("aime24_inplace_no_answer_prefix.jsonnet", "aime24", "test"),
        ("aime25_inplace_no_answer_prefix.jsonnet", "aime25", "test"),
        ("amc23_inplace_no_answer_prefix.jsonnet", "amc23", "test"),
        (
            "olympiadbench_hf_inplace_no_answer_prefix.jsonnet",
            "olympiadbench_hf",
            "train",
        ),
        ("collegeMath_inplace_no_answer_prefix.jsonnet", "collegeMath", "test"),
    ),
)
def test_local_eval_dataset_builds_task(config_name, dataset_path, split, monkeypatch):
    if not (DATA / dataset_path / "dataset_dict.json").exists():
        pytest.skip(f"data/{dataset_path} has not been downloaded")

    monkeypatch.chdir(ROOT)
    config_json = _jsonnet.evaluate_file(str(CONFIGS / "tasks" / config_name))
    task = Task.from_params(Params(json.loads(config_json)))
    dataset = task.get_datasets(split)

    assert len(dataset) > 0
    assert {"_treetune__idx", "problem", "answer", "query"} <= set(
        dataset.column_names
    )
    assert dataset[0]["query"] == dataset[0]["problem"]
    assert dataset[0]["answer"] not in (None, "", [])


@pytest.mark.parametrize(
    "config_name",
    (
        "deepseekR1Qwen_for_MATH_eval.jsonnet",
        "qwen1_5b_base_for_MATH_eval.jsonnet",
        "sft_deepseekmath_for_MATH_eval.jsonnet",
        "sft_rho1b_for_MATH_eval.jsonnet",
    ),
)
def test_math_eval_configs_compile_with_all_benchmarks(config_name):
    config = json.loads(_jsonnet.evaluate_file(str(CONFIGS / config_name)))
    inference_names = {
        pipeline["inference_name"] for pipeline in config["inference_pipelines"]
    }
    assert {
        "math_test",
        "aime24_test",
        "aime25_test",
        "amc23_test",
        "olympiadbench_test",
        "collegeMath_test",
    } <= inference_names


def test_smollm_eval_config_uses_one_model_and_small_gpu_limits():
    config = json.loads(
        _jsonnet.evaluate_file(
            str(CONFIGS / "polIter_smollm_135m_eval_MATH.jsonnet"),
            ext_vars={
                "APP_SEED": "42",
                "APP_DISABLE_FLASH_ATTENTION": "1",
            },
        )
    )
    model_name = "HuggingFaceTB/SmolLM2-135M"

    assert config["tokenizer"]["hf_model_name"] == model_name
    assert config["evaluation_vllm_server"]["max_num_seqs"] == 8
    assert config["evaluation_vllm_server"]["max_model_len"] == 1024

    for pipeline in config["inference_pipelines"]:
        strategy = pipeline["inference_strategy"]
        assert strategy["guidance_llm"]["model"] == model_name
        assert strategy["guidance_llm"]["tokenizer_name"] == model_name
        assert (
            strategy["node_expander"]["tokenizer"]["hf_model_name"]
            == model_name
        )
        assert strategy["node_expander"]["model_context_size"] == 1024


def test_eval_pipeline_selector_filters_requested_datasets():
    config = json.loads(
        _jsonnet.evaluate_snippet(
            "snippet",
            (
                f'(import "{CONFIGS / "deepseekR1Qwen_for_MATH_eval.jsonnet"}")'
                f' + (import "{CONFIGS / "evaluation" / "select_pipelines.jsonnet"}")'
            ),
            ext_vars={"APP_EVAL_PIPELINES": "math_test,aime24_test"},
        )
    )

    assert [
        pipeline["inference_name"] for pipeline in config["inference_pipelines"]
    ] == ["math_test", "aime24_test"]


def test_eval_pipeline_selector_rejects_unknown_pipeline():
    with pytest.raises(RuntimeError, match="Unknown evaluation pipeline"):
        _jsonnet.evaluate_snippet(
            "snippet",
            (
                f'(import "{CONFIGS / "deepseekR1Qwen_for_MATH_eval.jsonnet"}")'
                f' + (import "{CONFIGS / "evaluation" / "select_pipelines.jsonnet"}")'
            ),
            ext_vars={"APP_EVAL_PIPELINES": "not_a_real_pipeline"},
        )


def test_eval_overrides_merge_before_pipeline_selector():
    config = json.loads(
        _jsonnet.evaluate_snippet(
            "snippet",
            (
                f'(import "{CONFIGS / "polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet"}")'
                f' + (import "{CONFIGS / "local" / "math_local_10.jsonnet"}")'
                f' + (import "{CONFIGS / "evaluation" / "select_pipelines.jsonnet"}")'
            ),
            ext_vars={
                "APP_SEED": "42",
                "APP_DISABLE_FLASH_ATTENTION": "0",
                "APP_EVAL_PIPELINES": "math_test",
            },
        )
    )

    assert [
        pipeline["inference_name"] for pipeline in config["inference_pipelines"]
    ] == ["math_test"]
    assert (
        config["inference_pipelines"][0]["task"]["dataset_dict_path"]
        == "data/math-local-10"
    )


def test_eval_cli_overrides_tokenizer_context_and_generation_limit():
    config = json.loads(
        _jsonnet.evaluate_snippet(
            "snippet",
            (
                f'(import "{CONFIGS / "deepseekR1Qwen_for_MATH_eval.jsonnet"}")'
                f' + (import "{CONFIGS / "evaluation" / "cli_overrides.jsonnet"}")'
            ),
            ext_vars={
                "APP_EVAL_TOKENIZER": "HuggingFaceTB/SmolLM2-135M",
                "APP_EVAL_CONTEXT_LENGTH": "4096",
                "APP_EVAL_MAX_NEW_TOKENS": "1024",
            },
        )
    )

    assert (
        config["tokenizer"]["hf_model_name"]
        == "HuggingFaceTB/SmolLM2-135M"
    )
    assert config["evaluation_vllm_server"]["max_model_len"] == 4096
    for pipeline in config["inference_pipelines"]:
        strategy = pipeline["inference_strategy"]
        assert (
            strategy["guidance_llm"]["tokenizer_name"]
            == "HuggingFaceTB/SmolLM2-135M"
        )
        assert (
            strategy["node_expander"]["tokenizer"]["hf_model_name"]
            == "HuggingFaceTB/SmolLM2-135M"
        )
        assert strategy["node_expander"]["model_context_size"] == 4096
        assert strategy["node_expander"]["program_kwargs"]["max_tokens"] == 1024


def test_evaluate_script_selects_datasets_and_preserves_extra_args(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_deepspeed = fake_bin / "deepspeed"
    fake_deepspeed.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"APP_EVAL_PIPELINES=${APP_EVAL_PIPELINES:-}\"\n"
        "echo \"APP_EVAL_TOKENIZER=${APP_EVAL_TOKENIZER:-}\"\n"
        "echo \"APP_EVAL_CONTEXT_LENGTH=${APP_EVAL_CONTEXT_LENGTH:-}\"\n"
        "echo \"APP_EVAL_MAX_NEW_TOKENS=${APP_EVAL_MAX_NEW_TOKENS:-}\"\n"
        "echo \"APP_EXPERIMENT_NAME=${APP_EXPERIMENT_NAME:-}\"\n"
        "printf 'ARG=%s\\n' \"$@\"\n"
    )
    fake_deepspeed.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["APP_DIRECTORY"] = str(tmp_path / "experiments")
    env["APP_EXPERIMENT_NAME"] = "test-eval-selection"

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "evaluate.sh"),
            (
                "polIter_deepseekR1Qwen_gear_tree_MATH,"
                "local/math_local_10"
            ),
            "/tmp/checkpoint/hf_pretrained",
            "--config",
            "configs/local/math_local_runtime.jsonnet",
            "--dataset",
            "math",
            "aime24",
            "--datasets",
            "olympiadbench",
            "--debug_mode=true",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (
        "Selected evaluation pipelines: "
        "math_test,aime24_test,olympiadbench_test"
    ) in result.stdout
    assert (
        "APP_EVAL_PIPELINES=math_test,aime24_test,olympiadbench_test"
    ) in result.stdout
    assert "select_pipelines.jsonnet" in result.stdout
    config_arg = next(
        line.removeprefix("ARG=")
        for line in result.stdout.splitlines()
        if "polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet" in line
    )
    assert config_arg.index(
        "polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet"
    ) < config_arg.index("local/math_local_10.jsonnet")
    assert config_arg.index(
        "local/math_local_10.jsonnet"
    ) < config_arg.index("local/math_local_runtime.jsonnet")
    assert config_arg.index(
        "local/math_local_runtime.jsonnet"
    ) < config_arg.index("evaluation/select_pipelines.jsonnet")
    assert "ARG=--debug_mode=true" in result.stdout
    assert "ARG=--last_policy_path" in result.stdout
    assert "ARG=/tmp/checkpoint/hf_pretrained" in result.stdout

    default_result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "evaluate.sh"),
            "polIter_deepseekR1Qwen_gear_tree_MATH",
            "/tmp/checkpoint/hf_pretrained",
            "--debug_mode=true",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "APP_EVAL_PIPELINES=" in default_result.stdout
    assert "Selected evaluation pipelines:" not in default_result.stdout
    assert "select_pipelines.jsonnet" not in default_result.stdout


def test_evaluate_script_runs_all_experiment_checkpoints_sequentially(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_deepspeed = fake_bin / "deepspeed"
    fake_deepspeed.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"APP_EXPERIMENT_NAME=${APP_EXPERIMENT_NAME:-}\"\n"
        "echo \"APP_EVAL_TOKENIZER=${APP_EVAL_TOKENIZER:-}\"\n"
        "echo \"APP_EVAL_CONTEXT_LENGTH=${APP_EVAL_CONTEXT_LENGTH:-}\"\n"
        "echo \"APP_EVAL_MAX_NEW_TOKENS=${APP_EVAL_MAX_NEW_TOKENS:-}\"\n"
        "printf 'ARG=%s\\n' \"$@\"\n"
    )
    fake_deepspeed.chmod(0o755)

    experiment = tmp_path / "training-run"
    first = experiment / "checkpoints" / "ckpt--iter_0001" / "hf_pretrained"
    second = experiment / "checkpoints" / "ckpt--iter_0010" / "hf_pretrained"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["APP_DIRECTORY"] = str(tmp_path / "eval-results")
    env["APP_EXPERIMENT_NAME"] = "checkpoint-sweep"

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "evaluate.sh"),
            "polIter_deepseekR1Qwen_gear_tree_MATH",
            str(experiment),
            "--all-checkpoints",
            "--tokenizer",
            "HuggingFaceTB/SmolLM2-135M",
            "--context-length",
            "4096",
            "--max-new-tokens",
            "1024",
            "--dataset",
            "aime24",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.count("Evaluating checkpoint ") == 2
    assert str(first) in result.stdout
    assert str(second) in result.stdout
    assert "APP_EXPERIMENT_NAME=checkpoint-sweep-001-ckpt--iter_0001" in result.stdout
    assert "APP_EXPERIMENT_NAME=checkpoint-sweep-002-ckpt--iter_0010" in result.stdout
    assert result.stdout.count(
        "APP_EVAL_TOKENIZER=HuggingFaceTB/SmolLM2-135M"
    ) == 2
    assert result.stdout.count("APP_EVAL_CONTEXT_LENGTH=4096") == 2
    assert result.stdout.count("APP_EVAL_MAX_NEW_TOKENS=1024") == 2
    assert result.stdout.count("cli_overrides.jsonnet") == 2
