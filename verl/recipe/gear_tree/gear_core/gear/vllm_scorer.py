"""Adapter that turns the SPO vLLM api_base into an `LPScorer`.

vLLM exposes the OpenAI-style `/v1/completions` endpoint.  Setting
`echo=True, logprobs=1, max_tokens=0` makes vLLM return per-token logprobs
for every token in the prompt — exactly what we need to compute
`log pi(y_i | traj(s))`: tokenize the prefix once, then for each y_i send
`prefix + y_i` and sum the logprobs of the tail tokens.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .lp_scorer import LPScorer

logger = logging.getLogger(__name__)


@dataclass
class VLLMLogprobClient:
    api_base: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 120.0
    max_concurrency: int = 64
    retry_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    _semaphore: Optional[asyncio.Semaphore] = None
    _client: Optional[Any] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None

    async def _ensure_async_resources(self) -> None:
        """Create asyncio/httpx resources inside the currently running loop.

        Creating asyncio primitives or async HTTP clients before a loop is
        running can later surface as ``AttributeError: 'NoneType' object has no
        attribute 'create_future'`` from ``asyncio`` internals.  The inference
        strategy may also be reused across calls that are wrapped by separate
        ``asyncio.run(...)`` loops, so recreate resources if the active loop
        changes.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "VLLMLogprobClient.prompt_logprobs must be awaited inside a running asyncio loop"
            ) from exc

        if self._client is not None and self._loop is loop:
            return

        if self._client is not None:
            try:
                await self._client.aclose()
            except RuntimeError:
                # The previous client may belong to an already-closed loop.
                # Dropping it is safer than trying to reuse stale loop-bound
                # resources.
                pass

        self._loop = loop
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._semaphore = None
        self._loop = None

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.WriteError,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ),
        ):
            return True
        return isinstance(exc, httpx.HTTPStatusError) and (
            exc.response.status_code == 429 or exc.response.status_code >= 500
        )


    @staticmethod
    def _entropy_from_top_logprobs(top_logprobs: Dict[str, float]) -> float:
        if not top_logprobs:
            return 0.0
        probs = []
        for value in top_logprobs.values():
            try:
                lp = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(lp):
                probs.append(math.exp(lp))
        total = sum(probs)
        if total <= 0.0:
            return 0.0
        return float(-sum((p / total) * math.log(p / total) for p in probs if p > 0.0))

    async def completion_with_token_entropies(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.7,
        top_logprobs: int = 5,
    ) -> Tuple[str, List[float], List[str]]:
        """Generate text and return generated-token entropy details."""

        url = f"{self.api_base.rstrip('/')}/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max(int(max_tokens), 1),
            "logprobs": max(int(top_logprobs), 1),
            "echo": False,
            "temperature": float(temperature),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        await self._ensure_async_resources()
        assert self._semaphore is not None
        assert self._client is not None

        attempts = max(1, int(self.retry_attempts))
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    resp = await self._client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                break
            except Exception as exc:
                retryable = self._is_retryable_error(exc)
                if retryable and attempt < attempts:
                    delay = max(0.0, self.retry_backoff_seconds) * (2 ** (attempt - 1))
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                if isinstance(exc, httpx.HTTPStatusError):
                    raise RuntimeError(
                        "vLLM entropy request failed with HTTP "
                        f"{exc.response.status_code} for url={url!r}, "
                        f"model={self.model!r} after {attempt} attempt(s): "
                        f"{exc.response.text[:500]}"
                    ) from exc
                if retryable:
                    raise RuntimeError(
                        f"vLLM entropy connection failed for url={url!r}, "
                        f"model={self.model!r} after {attempt} attempt(s): {exc!r}"
                    ) from exc
                raise

        choice = data["choices"][0]
        text = choice.get("text", "")
        logprobs = choice.get("logprobs", {}) or {}
        top_rows = logprobs.get("top_logprobs") or []
        tokens = logprobs.get("tokens") or []
        entropies = [self._entropy_from_top_logprobs(row or {}) for row in top_rows]
        return text, entropies, list(tokens)

    async def prompt_logprobs(self, prompt: str) -> List[float]:
        """Return per-token prompt logprobs, retrying transient vLLM failures."""

        url = f"{self.api_base.rstrip('/')}/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        await self._ensure_async_resources()
        assert self._semaphore is not None
        assert self._client is not None

        attempts = max(1, int(self.retry_attempts))
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    resp = await self._client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                break
            except Exception as exc:
                retryable = self._is_retryable_error(exc)
                if retryable and attempt < attempts:
                    delay = max(0.0, self.retry_backoff_seconds) * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient vLLM logprob failure for url=%r, model=%r "
                        "(attempt %d/%d); retrying in %.2fs: %r",
                        url,
                        self.model,
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue

                if isinstance(exc, httpx.HTTPStatusError):
                    response_text = exc.response.text[:500]
                    raise RuntimeError(
                        "vLLM logprob request failed with HTTP "
                        f"{exc.response.status_code} for url={url!r}, "
                        f"model={self.model!r} after {attempt} attempt(s): "
                        f"{response_text}"
                    ) from exc
                if retryable:
                    raise RuntimeError(
                        f"vLLM logprob connection failed for url={url!r}, "
                        f"model={self.model!r} after {attempt} attempt(s): {exc!r}"
                    ) from exc
                raise

        choice = data["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs") or []
        return list(token_logprobs)


def make_lp_scorer(client: VLLMLogprobClient, tokenize_fn) -> LPScorer:
    async def score_fn(prompt: str, **_):
        return await client.prompt_logprobs(prompt)

    return LPScorer(score_fn=score_fn, tokenize_fn=tokenize_fn)
