from __future__ import annotations

import logging
import time
from typing import AsyncIterator

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from helix.models.request import HelixRequest, Message, Priority, UserTier

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response schemas (OpenAI-compatible) ────────────

class ChatCompletionRequest(BaseModel):
    model: str = "llama3"
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True
    # Helix extensions (optional — ignored by standard OpenAI clients)
    session_id: str | None = None
    priority: str | None = None        # "low" | "normal" | "high" | "critical"
    user_tier: str | None = None       # "free" | "standard" | "premium"
    deadline_seconds: float | None = None


def _build_helix_request(
    body: ChatCompletionRequest,
    client_id: str,
) -> HelixRequest:
    """Convert OpenAI-style request body into a HelixRequest."""
    messages = [
        Message(role=m["role"], content=m["content"])
        for m in body.messages
    ]

    priority_map = {
        "low": Priority.LOW,
        "normal": Priority.NORMAL,
        "high": Priority.HIGH,
        "critical": Priority.CRITICAL,
    }
    tier_map = {
        "free": UserTier.FREE,
        "standard": UserTier.STANDARD,
        "premium": UserTier.PREMIUM,
    }

    deadline_ms = (
        (time.time() + body.deadline_seconds) * 1000
        if body.deadline_seconds
        else (time.time() + 30) * 1000
    )

    return HelixRequest(
        session_id=body.session_id or None,
        client_id=client_id,
        model=body.model,
        messages=messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=body.stream,
        priority=priority_map.get(body.priority or "normal", Priority.NORMAL),
        user_tier=tier_map.get(body.user_tier or "standard", UserTier.STANDARD),
        deadline_ms=deadline_ms,
    )


def _sse_chunk(content: str, model: str) -> bytes:
    """Format a token as an OpenAI-compatible SSE data chunk."""
    chunk = {
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "delta": {"content": content},
                "index": 0,
                "finish_reason": None,
            }
        ],
    }
    return b"data: " + orjson.dumps(chunk) + b"\n\n"


def _sse_done() -> bytes:
    """SSE stream terminator."""
    return b"data: [DONE]\n\n"


async def _stream_response(
    request: HelixRequest,
    pool: object,
    model: str,
) -> AsyncIterator[bytes]:
    """
    Pull tokens from the worker pool and yield SSE-formatted bytes.
    Sends [DONE] sentinel at the end.
    """
    try:
        async for token in pool.generate(request):  # type: ignore[attr-defined]
            yield _sse_chunk(token, model)
        yield _sse_done()
    except Exception as exc:
        logger.error(
            "Stream error for request %s: %s", request.request_id, exc
        )
        error_chunk = orjson.dumps({"error": str(exc)})
        yield b"data: " + error_chunk + b"\n\n"
        yield _sse_done()


# ── Routes ────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.
    Always streams — non-streaming support comes in Phase 2.
    """
    pool = request.app.state.worker_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Worker pool not initialized")

    client_id = request.headers.get("X-Client-ID", "anonymous")
    helix_req = _build_helix_request(body, client_id)

    logger.info(
        "Incoming request id=%s model=%s client=%s",
        helix_req.request_id,
        helix_req.model,
        client_id,
    )

    return StreamingResponse(
        _stream_response(helix_req, pool, body.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": helix_req.request_id,
            "X-Session-ID": helix_req.session_id,
        },
    )


@router.get("/v1/models")
async def list_models(request: Request) -> dict:
    """List all models available across registered workers."""
    registry = request.app.state.registry
    workers = registry.all_workers() if registry else []
    model_set: set[str] = set()
    for w in workers:
        model_set.update(w.supported_models)
    if not model_set:
        model_set.add("llama3")  # Default assumption

    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "helix"}
            for m in sorted(model_set)
        ],
    }


@router.get("/health")
async def health(request: Request) -> dict:
    """Basic liveness check — always returns 200 if server is up."""
    pool = request.app.state.worker_pool
    registry = request.app.state.registry
    healthy = (
        len(registry.get_healthy_workers()) if registry else 0
    )
    return {
        "status": "ok",
        "healthy_workers": healthy,
        "pool_stats": pool.pool_stats() if pool else {},
    }


@router.get("/v1/nodes")
async def list_nodes(request: Request) -> dict:
    """Helix-specific: show all registered workers and their status."""
    registry = request.app.state.registry
    if not registry:
        return {"workers": []}
    return registry.status_summary()
