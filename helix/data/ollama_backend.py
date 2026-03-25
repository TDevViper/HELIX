from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import httpx
import orjson

from helix.data.backend_abc import LLMBackend
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerStats


class OllamaBackend(LLMBackend):
    """
    Adapter for Ollama — the simplest backend, used for local testing.
    Talks to Ollama's native /api/chat endpoint with streaming NDJSON.
    """

    def __init__(
        self,
        worker_id: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
    ) -> None:
        self._worker_id = worker_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._memory_used_gb: float = 0.0
        self._client: httpx.AsyncClient | None = None

    # ── Identity ──────────────────────────────────────────────

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

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
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                f"OllamaBackend {self._worker_id} not started. "
                "Call startup() first."
            )
        return self._client

    # ── Core inference ────────────────────────────────────────

    async def generate(self, request: HelixRequest) -> AsyncIterator[str]:
        """
        Stream tokens from Ollama /api/chat.
        Ollama returns NDJSON: one JSON object per line.
        Each line has {"message": {"content": "<token>"}, "done": false}
        """
        client = self._get_client()
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with client.stream(
            "POST",
            "/api/chat",
            content=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"Ollama error {response.status_code}: {body.decode()}"
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

                if chunk.get("done", False):
                    break

    # ── Health & stats ────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Hits GET /api/tags — Ollama's lightest endpoint.
        Returns False on any error, never raises.
        """
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get("/api/tags"),
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def get_stats(self) -> WorkerStats:
        """
        Ollama has no native metrics endpoint, so we approximate.
        Memory usage is read from /api/ps (process status) if available.
        Falls back to zeroed stats — never raises.
        """
        stats = WorkerStats(worker_id=self._worker_id)
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get("/api/ps"),
                timeout=2.0,
            )
            if response.status_code == 200:
                data = orjson.loads(response.content)
                models = data.get("models", [])
                total_vram = sum(
                    m.get("size_vram", 0) for m in models
                )
                self._memory_used_gb = total_vram / (1024 ** 3)
                stats.memory_used_gb = self._memory_used_gb
                stats.memory_total_gb = 0.0  # Ollama does not report total
        except Exception:
            pass  # Return zeroed stats on any failure
        return stats
