import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_with_fake_python(tmp_path, script, *args, dtype=None):
    capture_path = tmp_path / "python_args.txt"
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n'
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["CAPTURE_FILE"] = str(capture_path)
    env["HF_HOME"] = str(tmp_path / "hf")
    if dtype is None:
        env.pop("VLLM_DTYPE", None)
    else:
        env["VLLM_DTYPE"] = dtype

    subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *args],
        check=True,
        env=env,
    )
    return capture_path.read_text().splitlines()


def _dtype_arg(args):
    return args[args.index("--dtype") + 1]


def test_positional_launcher_defaults_empty_dtype_to_bfloat16(tmp_path):
    args = _run_with_fake_python(
        tmp_path,
        "start_vllm_server.sh",
        "model",
        "8000",
        "42",
        "8",
        "3",
        dtype="",
    )

    assert _dtype_arg(args) == "bfloat16"


def test_named_launcher_defaults_empty_dtype_to_bfloat16(tmp_path):
    args = _run_with_fake_python(
        tmp_path,
        "start_vllm_server_named_params.sh",
        "--model",
        "model",
        "--port",
        "8000",
        "--seed",
        "42",
        "--gpu-idx",
        "3",
        dtype="",
    )

    assert _dtype_arg(args) == "bfloat16"


def test_named_launcher_defaults_unset_dtype_to_bfloat16(tmp_path):
    args = _run_with_fake_python(
        tmp_path,
        "start_vllm_server_named_params.sh",
        "--model",
        "model",
        "--port",
        "8000",
        "--seed",
        "42",
    )

    assert _dtype_arg(args) == "bfloat16"


def test_named_launcher_respects_dtype_override(tmp_path):
    args = _run_with_fake_python(
        tmp_path,
        "start_vllm_server_named_params.sh",
        "--model",
        "model",
        "--port",
        "8000",
        "--seed",
        "42",
        dtype="bfloat16",
    )

    assert _dtype_arg(args) == "bfloat16"
