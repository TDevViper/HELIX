from __future__ import annotations

import logging
from typing import AsyncIterator

import orjson

from helix.models.request import HelixRequest

logger = logging.getLogger(__name__)


def sse_chunk(content: str, model: str) -> bytes:
    chunk = {
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}],
    }
    return b"data: " + orjson.dumps(chunk) + b"\n\n"


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


async def stream_from_pool(
    request: HelixRequest,
    pool: object,
    model: str,
) -> AsyncIterator[bytes]:
    """
    Pulls tokens from worker pool and yields SSE bytes.
    On worker failure: logs error, sends error chunk, always sends [DONE].
    Never drops the [DONE] sentinel — clients depend on it to close the stream.
    """
    try:
        async for token in pool.generate(request):  # type: ignore[attr-defined]
            yield sse_chunk(token, model)
        yield sse_done()
    except Exception as exc:
        logger.error(
            "Stream error request=%s worker=%s: %s",
            request.request_id,
            request.assigned_worker_id,
            exc,
        )
        error_payload = orjson.dumps({"error": {"message": str(exc), "type": "worker_error"}})
        yield b"data: " + error_payload + b"\n\n"
        yield sse_done()
