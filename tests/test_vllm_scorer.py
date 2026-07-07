import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")
pytest.importorskip("numpy")

from treetune.gear import vllm_scorer


class FakeHTTPError(Exception):
    pass


class FakeHTTPStatusError(FakeHTTPError):
    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class FakeReadError(FakeHTTPError):
    pass


class FakeResponse:
    def __init__(self, status_code=200, *, data=None, text=""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTPStatusError(self)

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.attempts = 0

    async def post(self, *_args, **_kwargs):
        self.attempts += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        return None


FAKE_HTTPX = SimpleNamespace(
    ConnectError=FakeReadError,
    ConnectTimeout=FakeReadError,
    ReadError=FakeReadError,
    ReadTimeout=FakeReadError,
    WriteError=FakeReadError,
    WriteTimeout=FakeReadError,
    PoolTimeout=FakeReadError,
    HTTPStatusError=FakeHTTPStatusError,
)


def _run_with_outcomes(monkeypatch, outcomes, *, retry_attempts=3):
    monkeypatch.setattr(vllm_scorer, "httpx", FAKE_HTTPX)

    async def run():
        client = vllm_scorer.VLLMLogprobClient(
            api_base="http://vllm.test/v1",
            model="test-model",
            retry_attempts=retry_attempts,
            retry_backoff_seconds=0.0,
        )
        fake_client = FakeAsyncClient(outcomes)
        client._loop = asyncio.get_running_loop()
        client._semaphore = asyncio.Semaphore(client.max_concurrency)
        client._client = fake_client
        try:
            result = await client.prompt_logprobs("prompt")
            return result, fake_client.attempts
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_prompt_logprobs_retries_transient_read_error(monkeypatch):
    success = FakeResponse(
        data={
            "choices": [{"logprobs": {"token_logprobs": [None, -0.2, -0.3]}}]
        }
    )

    result, attempts = _run_with_outcomes(
        monkeypatch,
        [FakeReadError("server disconnected"), FakeReadError("again"), success],
    )

    assert result == [None, -0.2, -0.3]
    assert attempts == 3


def test_prompt_logprobs_reports_exhausted_retry_count(monkeypatch):
    with pytest.raises(RuntimeError, match=r"after 2 attempt\(s\)"):
        _run_with_outcomes(
            monkeypatch,
            [FakeReadError("server disconnected"), FakeReadError("again")],
            retry_attempts=2,
        )


def test_prompt_logprobs_does_not_retry_non_transient_http_error(monkeypatch):
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _run_with_outcomes(monkeypatch, [FakeResponse(400, text="bad request")])


def test_completion_with_token_entropies_parses_top_logprobs(monkeypatch):
    monkeypatch.setattr(vllm_scorer, "httpx", FAKE_HTTPX)

    async def run():
        client = vllm_scorer.VLLMLogprobClient(
            api_base="http://vllm.test/v1",
            model="test-model",
            retry_backoff_seconds=0.0,
        )
        fake_client = FakeAsyncClient([
            FakeResponse(
                data={
                    "choices": [
                        {
                            "text": " ab",
                            "logprobs": {
                                "top_logprobs": [
                                    {"a": -0.1, "b": -2.0},
                                    {"c": -0.7, "d": -0.7},
                                ],
                                "tokens": [" a", "b"],
                            },
                        }
                    ]
                }
            )
        ])
        client._loop = asyncio.get_running_loop()
        client._semaphore = asyncio.Semaphore(client.max_concurrency)
        client._client = fake_client
        try:
            return await client.completion_with_token_entropies("prompt", max_tokens=2)
        finally:
            await client.aclose()

    text_out, entropies, tokens = asyncio.run(run())

    assert text_out == " ab"
    assert tokens == [" a", "b"]
    assert len(entropies) == 2
    assert entropies[1] > entropies[0]
