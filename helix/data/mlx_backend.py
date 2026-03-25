from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import orjson

from helix.data.backend_abc import LLMBackend
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerStats


class MLXBackend(LLMBackend):
    """
    Adapter for MLX-Serve — your existing Apple Silicon inference server.
    MLX-Serve exposes an OpenAI-compatible API, so we use /v1/chat/completions
    with SSE streaming (text/event-stream), not Ollama-style NDJSON.
    """

    def __init__(
        self,
        worker_id: str,
        base_url: str = "http://localhost:8001",
        timeout_seconds: int = 120,
        api_key: str = "mlx-local",
    ) -> None:
        self._worker_id = worker_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._api_key = api_key
        self._memory_used_gb: float = 0.0
        self._client: httpx.AsyncClient | None = None

    # ── Identity ──────────────────────────────────────────────

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def backend_type(self) -> BackendType:
        return BackendType.MLX

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def memory_used_gb(self) -> float:
        return self._memory_used_gb

    # ── Lifecycle ─────────────────────────────────────────────

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=self._timeout,
                write=10.0,
                pool=5.0,
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                f"MLXBackend {self._worker_id} not started. "
                "Call startup() first."
            )
        return self._client

    # ── Core inference ────────────────────────────────────────

    async def generate(self, request: HelixRequest) -> AsyncIterator[str]:
        """
        Stream tokens from MLX-Serve /v1/chat/completions.
        MLX-Serve returns OpenAI-compatible SSE:
          data: {"choices": [{"delta": {"content": "<token>"}}]}
          data: [DONE]
        """
        client = self._get_client()
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            content=orjson.dumps(payload),
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"MLX-Serve error {response.status_code}: {body.decode()}"
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break

                try:
                    chunk = orjson.loads(payload_str)
                except orjson.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    # ── Health & stats ────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Hits GET /health — MLX-Serve's health endpoint.
        Returns False on any error, never raises.
        """
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get("/health"),
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def get_stats(self) -> WorkerStats:
        """
        Fetches metrics from MLX-Serve /v1/stats endpoint.
        Falls back to zeroed stats — never raises.
        """
        stats = WorkerStats(worker_id=self._worker_id)
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get("/v1/stats"),
                timeout=2.0,
            )
            if response.status_code == 200:
                data = orjson.loads(response.content)
                memory_gb = data.get("memory_used_gb", 0.0)
                self._memory_used_gb = memory_gb
                stats.memory_used_gb = memory_gb
                stats.memory_total_gb = data.get("memory_total_gb", 0.0)
                stats.tokens_per_second = data.get("tokens_per_second", 0.0)
                stats.requests_active = data.get("active_requests", 0)
                stats.queue_depth = data.get("queue_depth", 0)
        except Exception:
            pass
        return stats
