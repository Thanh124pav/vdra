from types import SimpleNamespace

from treetune.common.vllm_server import VLLMServer


def test_force_free_gpu_memory_kills_only_vllm_processes_on_gpu(monkeypatch):
    server = VLLMServer.__new__(VLLMServer)
    commands = {
        101: "python -m vllm.entrypoints.openai.api_server",
        202: "python train.py",
        404: "python -m vllm.entrypoints.openai.api_server",
    }
    inspected_pids = []
    killed = []

    def fake_run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(stdout="101\n202\n404\n")

    def fake_pid_command(pid):
        inspected_pids.append(pid)
        return commands[pid]

    monkeypatch.setattr(
        "treetune.common.vllm_server.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(server, "_pid_command", fake_pid_command)
    monkeypatch.setattr(
        "treetune.common.vllm_server.os.kill",
        lambda pid, signal: killed.append(pid),
    )
    monkeypatch.setattr("treetune.common.vllm_server.time.sleep", lambda _: None)

    server._force_free_gpu_memory(gpu_idx=0, owned_pids={101, 202})

    assert set(inspected_pids) == {101, 202, 404}
    assert killed == [101]
