import json
import time
import uuid
from collections.abc import AsyncIterator


async def openai_sse_stream(
    chosen: str,
    model: str,
    *,
    empty_fallback: str | None = "Summary complete.",
) -> AsyncIterator[bytes]:
    """OpenAI-style chat.completion chunks for browsers / fetch streaming parsers."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    raw = str(chosen or "").strip()
    if raw:
        display = raw
    elif empty_fallback is None:
        display = ""
    else:
        display = empty_fallback
    base = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model}

    def pack(delta: dict, finish_reason: str | None) -> str:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield pack({"role": "assistant"}, None).encode("utf-8")
    yield pack({"content": display}, None).encode("utf-8")
    yield pack({}, "stop").encode("utf-8")
    yield b"data: [DONE]\n\n"
