from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import orjson

from helix.data.backend_abc import LLMBackend
from helix.models.request import HelixRequest
from helix.models.worker import BackendType, WorkerStats


class VLLMBackend(LLMBackend):
    """
    Adapter for vLLM — OpenAI-compatible inference server for CUDA GPUs.
    vLLM exposes /v1/chat/completions (SSE) and /metrics (Prometheus).
    The SSE parsing is identical to MLXBackend — both speak OpenAI protocol.
    vLLM-specific: we also read /metrics for detailed memory/throughput stats.
    """

    def __init__(
        self,
        worker_id: str,
        base_url: str = "http://localhost:8002",
        timeout_seconds: int = 120,
        api_key: str = "vllm-local",
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
        return BackendType.VLLM

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
                f"VLLMBackend {self._worker_id} not started. "
                "Call startup() first."
            )
        return self._client

    # ── Core inference ────────────────────────────────────────

    async def generate(self, request: HelixRequest) -> AsyncIterator[str]:
        """
        Stream tokens from vLLM /v1/chat/completions.
        Identical SSE format to MLX-Serve and OpenAI:
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
            # vLLM extras — passed through if present in extra_body
            **request.extra_body,
        }

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            content=orjson.dumps(payload),
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"vLLM error {response.status_code}: {body.decode()}"
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
        Hits GET /health — vLLM's standard health endpoint.
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
        Reads vLLM Prometheus metrics from /metrics endpoint.
        Parses text exposition format to extract key counters.
        Falls back to zeroed stats on any error — never raises.
        """
        stats = WorkerStats(worker_id=self._worker_id)
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get("/metrics"),
                timeout=2.0,
            )
            if response.status_code == 200:
                parsed = self._parse_prometheus_metrics(response.text)
                gpu_cache_usage = parsed.get(
                    "vllm:gpu_cache_usage_perc", 0.0
                )
                # vLLM reports cache as % of allocated GPU memory
                # We approximate memory_used from cache utilization
                self._memory_used_gb = gpu_cache_usage * 80.0  # assume 80GB GPU
                stats.memory_used_gb = self._memory_used_gb
                stats.memory_total_gb = 80.0
                stats.tokens_per_second = parsed.get(
                    "vllm:avg_generation_throughput_toks_per_s", 0.0
                )
                stats.requests_active = int(
                    parsed.get("vllm:num_requests_running", 0)
                )
                stats.queue_depth = int(
                    parsed.get("vllm:num_requests_waiting", 0)
                )
        except Exception:
            pass
        return stats

    def _parse_prometheus_metrics(self, text: str) -> dict[str, float]:
        """
        Parse Prometheus text exposition format into a flat dict.
        Skips comment lines (# HELP, # TYPE) and malformed lines.
        Example line: vllm:gpu_cache_usage_perc 0.42
        """
        result: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip labels: metric_name{label="val"} value → metric_name value
            if "{" in line:
                name = line[:line.index("{")]
                rest = line[line.index("}") + 1:].strip()
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                rest = parts[1]
            try:
                result[name] = float(rest.split()[0])
            except (ValueError, IndexError):
                continue
        return result
