import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..deps.common import create_completion_common_params
from ..schemas.models import OpenAIRequest
from ..services.backend import (
    call_litellm_optional,
    effective_api_key,
    load_vllm_config,
    merge_arm_config,
)
from ..services.messages import (
    is_brave_conversation_title_request,
    merge_completion_texts,
    resolve_completion_plan,
)
from ..services.storage import store_examples
from ..services.streaming import openai_sse_stream

v1_router = APIRouter()


@v1_router.post("/chat/completions", response_model=None)
async def v1_chat_completions(
    request: OpenAIRequest,
    background_tasks: BackgroundTasks,
    fastapi_response: Response,
    common: dict = Depends(create_completion_common_params),
) -> StreamingResponse | JSONResponse:
    """
    Up to four LLM completions (text chosen/rejected, image chosen/rejected). Writes one or two
    JSON files under output/ with shape { prompt, chosen, rejected } each.
    """
    if is_brave_conversation_title_request(request):
        if request.stream:
            return StreamingResponse(
                openai_sse_stream("", request.model, empty_fallback=None),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return JSONResponse(
            content={
                "id": "chatcmpl-ocelot",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "model": request.model,
            }
        )

    plan = resolve_completion_plan(request)
    config = load_vllm_config()
    api_base = config.get("api_base", "http://localhost:8000/v1")
    chosen_cfg = merge_arm_config(config, "chosen")
    rejected_cfg = merge_arm_config(config, "rejected")
    chosen_cfg["api_key"] = effective_api_key(config, chosen_cfg)
    rejected_cfg["api_key"] = effective_api_key(config, rejected_cfg)

    chosen_t, chosen_v, rej_t, rej_v = await asyncio.gather(
        call_litellm_optional(plan.chosen_text_msgs, api_base, chosen_cfg),
        call_litellm_optional(plan.chosen_vision_msgs, api_base, chosen_cfg),
        call_litellm_optional(plan.rejected_text_msgs, api_base, rejected_cfg),
        call_litellm_optional(plan.rejected_vision_msgs, api_base, rejected_cfg),
    )
    store_examples(plan, chosen_t, chosen_v, rej_t, rej_v)

    chosen_for_client = merge_completion_texts(chosen_t, chosen_v)

    if request.stream:
        return StreamingResponse(
            openai_sse_stream(chosen_for_client, request.model),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return JSONResponse(
        content={
            "id": "chatcmpl-ocelot",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": chosen_for_client},
                    "finish_reason": "stop",
                }
            ],
            "model": request.model,
        }
    )
