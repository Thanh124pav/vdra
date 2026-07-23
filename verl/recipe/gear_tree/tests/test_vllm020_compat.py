from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text()


def test_vllm020_lora_uses_lora_model_api():
    source = _read("verl/utils/vllm/utils.py")

    assert "from vllm.lora.lora_model import LoRAModel" in source
    assert "vllm.lora.models" not in source
    assert "model_vocab_size=self.vocab_size" in source
    assert "target_embedding_padding" not in source
    assert "embedding_modules=self.embedding_modules" not in source
    assert "embedding_padding_modules" not in source


def test_vllm020_async_server_uses_020_cli_and_log_api():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_async_server.py")

    assert "from vllm.utils.argparse_utils import FlexibleArgumentParser" in source
    assert "from vllm.utils import FlexibleArgumentParser" not in source
    assert "from vllm.utils import get_tcp_uri" not in source
    assert '"enable_log_requests"' in source
    assert "disable_log_requests" not in source
    assert "VDRA/VERL is tested against vLLM 0.20.x" in source


def test_vllm020_worker_wrapper_uses_v1_lazy_init_api():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")

    assert "from vllm.v1.worker.worker_base import WorkerWrapperBase" in source
    assert "from vllm.worker.worker_base import WorkerWrapperBase" not in source
    assert "WorkerWrapperBase(vllm_config=" not in source
    assert "self.inference_engine = WorkerWrapperBase()" in source


def test_vllm020_worker_dispatch_does_not_use_removed_execute_method():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")

    assert "self.inference_engine.execute_method" not in source
    assert "getattr(self.inference_engine, method)(*args, **kwargs)" in source
    assert "pickle.loads(method)(self.inference_engine, *args, **kwargs)" in source


def test_vllm020_rollout_uses_020_imports_without_newer_fallbacks():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")
    server_source = _read("verl/workers/rollout/vllm_rollout/vllm_async_server.py")

    assert "from vllm.config import CompilationConfig, LoRAConfig" in source
    assert "from vllm.config.compilation import CUDAGraphMode, CompilationMode" in source
    assert "CompilationLevel" not in source
    assert "mode=CompilationMode.VLLM_COMPILE" in source
    assert "cudagraph_mode=CUDAGraphMode.PIECEWISE" in source
    assert "cudagraph_capture_sizes=list(cudagraph_capture_sizes)" in source
    assert "vLLM >= 0.22" not in source
    assert "self.inference_engine.add_lora(lora_reqest)" in source
    assert "self.inference_engine.worker.add_lora" not in source
    assert "vllm.AsyncEngineArgs" not in server_source
    assert "AsyncEngineArgs.from_cli_args" in server_source


def test_vllm020_openai_app_state_uses_020_signature():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_async_server.py")

    assert "supported_tasks = await engine_client.get_supported_tasks()" in source
    assert "model_config = engine_client.model_config" in source
    assert "build_app(args, supported_tasks, model_config)" in source
    assert "init_app_state(engine_client, app.state, args, supported_tasks)" in source
    assert "init_app_state(engine_client, vllm_config, app.state, args)" not in source
