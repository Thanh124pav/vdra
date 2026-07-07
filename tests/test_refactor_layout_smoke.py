import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def _text_files(root: Path):
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            yield path


def test_refactor_layout_removes_legacy_spo_tree():
    assert not any((ROOT / "spo").rglob("*.*"))


def test_deepseek_r1_qwen_files_are_not_named_qwen1b():
    qwen1b_paths = [
        path
        for root in (CONFIGS, SCRIPTS)
        for path in _text_files(root)
        if "qwen1b" in path.as_posix()
    ]
    assert qwen1b_paths == []

    deepseek_model_paths = [
        path.relative_to(ROOT).as_posix()
        for root in (CONFIGS, SCRIPTS)
        for path in _text_files(root)
        if "DeepSeek-R1-Distill-Qwen-1.5B" in path.read_text()
    ]
    assert deepseek_model_paths
    assert all("deepseekR1Qwen" in path for path in deepseek_model_paths)


def test_deepseek_r1_qwen_7b_configs_exist_and_use_hf_model():
    hf_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    expected = [
        CONFIGS / "guidance_llms" / "deepseekR1Qwen7B.jsonnet",
        CONFIGS / "model_overrides" / "deepseekR1Qwen7B.jsonnet",
        CONFIGS / "deepseekR1Qwen7B_for_MATH_eval.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_spo_chain_MATH.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_spo_tree_MATH.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_gear_tree_MATH.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_grpo_MATH.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_grpo_point24.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_gear_tree_point24.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_spo_tree_point24.jsonnet",
        CONFIGS / "polIter_deepseekR1Qwen7B_spo_chain_point24.jsonnet",
        CONFIGS / "deepseekR1Qwen7B_for_point24_eval.jsonnet",
    ]
    assert all(path.is_file() for path in expected)
    assert hf_model in (CONFIGS / "model_overrides" / "deepseekR1Qwen7B.jsonnet").read_text()
    assert hf_model in (CONFIGS / "deepseekR1Qwen7B_for_MATH_eval.jsonnet").read_text()
    assert hf_model in (CONFIGS / "deepseekR1Qwen7B_for_point24_eval.jsonnet").read_text()


def test_config_imports_resolve_inside_unified_configs_tree():
    import_re = re.compile(r"import\s+'([^']+)'")
    missing = []
    legacy_spo_imports = []
    for path in _text_files(CONFIGS):
        text = "\n".join(line.split("//", 1)[0] for line in path.read_text().splitlines())
        for imported in import_re.findall(text):
            if "spo/configs" in imported:
                legacy_spo_imports.append((path.relative_to(ROOT).as_posix(), imported))
                continue
            candidates = [
                (path.parent / imported).resolve(),
                (CONFIGS / imported).resolve(),
                (ROOT / imported).resolve(),
            ]
            if not any(candidate.is_file() for candidate in candidates):
                missing.append((path.relative_to(ROOT).as_posix(), imported))

    assert legacy_spo_imports == []
    assert missing == []


def test_gear_defaults_expose_only_online_gear_knobs():
    defaults = (CONFIGS / "gear_defaults.libsonnet").read_text()
    overlay = (CONFIGS / "gear_overlay.libsonnet").read_text()

    assert "skip_near_leaf_expand: true" in defaults
    assert "n_min: 0" in defaults
    assert "score_retry_attempts: 5" in defaults
    assert "budget_queue_count: 4" in defaults
    assert "budget_queue_timeout_seconds: 1.0" in defaults
    assert "root_allocation: true" in defaults
    assert "score_retry_backoff_seconds: 0.5" in defaults
    assert "gear_k_algorithm: $.gear.k_algorithm" in overlay
    assert "gear_generation_mode: $.gear.generation_mode" in overlay
    assert "k_algorithm: 'hierarchical'" in defaults
    assert "generation_mode: 'single_request'" in defaults
    for removed in [
        "alpha:",
        "use_dkw:",
        "eta_override:",
        "enable_share:",
        "enable_prune:",
        "share_target:",
        "local_value_share:",
        "share_pair_budget_fraction:",
        "share_use_confidence:",
        "algorithm_mode:",
        "tv_estimator:",
    ]:
        assert removed not in defaults
    assert "gear_K:" not in overlay
    assert "gear_m:" not in overlay
    for removed in [
        "gear_alpha:",
        "gear_use_dkw:",
        "gear_eta_override:",
        "gear_enable_share:",
        "gear_enable_prune:",
        "gear_share_target:",
        "gear_local_value_share:",
        "gear_share_pair_budget_fraction:",
        "gear_share_use_confidence:",
        "gear_algorithm_mode:",
        "gear_tv_estimator:",
    ]:
        assert removed not in overlay
    assert "K: 10" not in defaults
    assert "m: 100" not in defaults
    assert "gear_n_min: $.gear.n_min" in overlay
    assert "gear_score_retry_attempts: $.gear.score_retry_attempts" in overlay
    assert (
        "gear_score_retry_backoff_seconds: $.gear.score_retry_backoff_seconds"
        in overlay
    )
    assert "store_logprobs" not in overlay
    assert "program_kwargs+:" in overlay
    assert "logprobs: 1" in overlay


def test_all_training_runs_disable_dataset_sampling_with_replacement_and_kl():
    common_script = (ROOT / "scripts" / "_common.sh").read_text()
    no_replacement_config = (
        ROOT / "configs" / "episode_generators" / "noSamplRplc.jsonnet"
    ).read_text()

    assert "noSamplRplc.jsonnet" in common_script
    assert 'resolved_cfgs+=",${no_sample_replacement_config}"' in common_script
    assert "dataset_sample_with_replacement: false" in no_replacement_config
    assert 'GEAR_KL_COEF:-0.0' in common_script
    assert "init_kl_coef: ${kl_coef}" in common_script


def test_common_script_can_override_online_gear_modes():
    common_script = (ROOT / "scripts" / "_common.sh").read_text()

    assert "GEAR_K_ALGORITHM" in common_script
    assert "GEAR_GENERATION_MODE" in common_script
    assert "gear_k_algorithm: '${GEAR_K_ALGORITHM}'" in common_script
    assert "gear_generation_mode: '${GEAR_GENERATION_MODE}'" in common_script
    assert "${GEAR_K:-}" not in common_script
    assert "${GEAR_M:-}" not in common_script
    assert "gear_K:" not in common_script
    assert "gear_m:" not in common_script
    assert "GEAR_TV_ESTIMATOR" not in common_script
    assert "GEAR_ALGORITHM_MODE" not in common_script
    assert "GEAR_SHARE_PAIR_BUDGET_FRACTION" not in common_script
    assert "gear_tv_estimator:" not in common_script
    assert "gear_algorithm_mode:" not in common_script
    assert "gear_share_pair_budget_fraction:" not in common_script


def test_common_script_default_eval_and_save_frequency_is_ten():
    common_script = (ROOT / "scripts" / "_common.sh").read_text()
    training_args = (CONFIGS / "trainers" / "training_args.jsonnet").read_text()

    assert 'GEAR_EVAL_EVERY_N_ITERATIONS:-10' in common_script
    assert 'GEAR_SAVE_STEPS:-10' in common_script
    assert "save_steps: 10" in training_args
