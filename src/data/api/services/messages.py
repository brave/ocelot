import copy
import json
from dataclasses import dataclass

from ..schemas.models import OpenAIRequest
from . import prompts

# Brave multimodal UI sends these as separate text parts alongside brave-page-text + screenshots.
_BRAVE_SCREENSHOT_UI_TEXT_NORMALIZED = frozenset(
    {
        "these images are screenshots",
        "summarise",
        "these images are screenshots summarise",
    }
)


def _normalized_text_part_body(part: dict) -> str:
    return " ".join((part.get("text") or "").split()).lower().strip()


def _is_brave_screenshot_ui_text(part: dict) -> bool:
    if part.get("type") != "text":
        return False
    return _normalized_text_part_body(part) in _BRAVE_SCREENSHOT_UI_TEXT_NORMALIZED


def _image_url_part_nonempty(part: dict) -> bool:
    if part.get("type") != "image_url":
        return False
    iu = part.get("image_url")
    if isinstance(iu, dict):
        return bool((iu.get("url") or "").strip())
    if isinstance(iu, str):
        return bool(iu.strip())
    return False


def _collect_image_url_parts(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and _image_url_part_nonempty(part):
                out.append(copy.deepcopy(part))
    return out


def _messages_without_image_parts(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in copy.deepcopy(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_parts: list = []
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "image_url":
                continue
            new_parts.append(part)
        msg["content"] = new_parts
        out.append(msg)
    return out


def _has_brave_page_text(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "brave-page-text":
                continue
            if (part.get("text") or "").strip():
                return True
    return False


def _vision_user_messages(image_parts: list[dict], instruction: str) -> list[dict]:
    if not image_parts:
        return []
    content = copy.deepcopy(image_parts)
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def _wrap_webpage_in_page_tags(messages: list[dict], summary_instruction: str) -> list[dict]:
    """
    Wrap brave-page-text blocks in <page>...</page> and append summary_instruction after them
    (same user message). Drops Brave multimodal UI text ("These images are screenshots", "Summarise").
    Other content parts are preserved in order.
    """
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts: list = []
        had_page = False
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "brave-request-summary":
                continue
            if part.get("type") == "brave-page-text":
                t = (part.get("text") or "").strip()
                if t:
                    new_parts.append({"type": "text", "text": f"<page>\n{t}\n</page>"})
                    had_page = True
                continue
            if _is_brave_screenshot_ui_text(part):
                continue
            new_parts.append(copy.deepcopy(part))
        if had_page:
            new_parts.append({"type": "text", "text": summary_instruction})
            msg["content"] = new_parts
    return out


def build_prompts(request: OpenAIRequest) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build training_prompt (stored), chosen_prompt (for chosen model), rejected_prompt (for rejected model).
    Webpage segments are wrapped in <page> tags; chosen/training use the detailed instruction,
    rejected uses a short minimal instruction.
    """
    base = [m for m in request.messages]
    training_prompt = _wrap_webpage_in_page_tags(base, prompts.CHOSEN_SUMMARY_INSTRUCTION)
    chosen_prompt = _wrap_webpage_in_page_tags(base, prompts.CHOSEN_SUMMARY_INSTRUCTION)
    rejected_prompt = _wrap_webpage_in_page_tags(base, prompts.REJECTED_SUMMARY_INSTRUCTION)
    return training_prompt, chosen_prompt, rejected_prompt


@dataclass
class CompletionPlan:
    """
    Four LLM calls when both modalities exist: text×(chosen,rejected) + image×(chosen,rejected).
    None means skip that call / that output file.
    """

    chosen_text_msgs: list[dict] | None
    rejected_text_msgs: list[dict] | None
    chosen_vision_msgs: list[dict] | None
    rejected_vision_msgs: list[dict] | None


def resolve_completion_plan(request: OpenAIRequest) -> CompletionPlan:
    images = _collect_image_url_parts(request.messages)
    text_only = _messages_without_image_parts(request.messages)
    has_page = _has_brave_page_text(text_only)

    chosen_vis = (
        _vision_user_messages(images, prompts.CHOSEN_SUMMARY_INSTRUCTION_VISION) if images else None
    )
    rejected_vis = (
        _vision_user_messages(images, prompts.REJECTED_SUMMARY_INSTRUCTION_VISION)
        if images
        else None
    )

    if not images:
        _, c, r = build_prompts(request)
        return CompletionPlan(c, r, None, None)

    if not has_page:
        return CompletionPlan(None, None, chosen_vis, rejected_vis)

    chosen_txt = _wrap_webpage_in_page_tags(
        copy.deepcopy(text_only), prompts.CHOSEN_SUMMARY_INSTRUCTION
    )
    rejected_txt = _wrap_webpage_in_page_tags(
        copy.deepcopy(text_only), prompts.REJECTED_SUMMARY_INSTRUCTION
    )
    return CompletionPlan(chosen_txt, rejected_txt, chosen_vis, rejected_vis)


def _normalize_content_part(part: dict) -> list[dict]:
    """Map one content block to OpenAI-style parts LiteLLM accepts (text / image_url)."""
    ptype = part.get("type") or ""
    if ptype == "text" and "text" in part:
        return [{"type": "text", "text": part["text"]}]
    if ptype == "image_url" and "image_url" in part:
        return [{"type": "image_url", "image_url": part["image_url"]}]
    if ptype == "brave-page-text":
        t = (part.get("text") or "").strip()
        return [{"type": "text", "text": t}] if t else []
    if ptype == "brave-request-summary":
        return []
    if ptype == "brave-user-memory":
        mem = part.get("memory")
        if isinstance(mem, dict):
            body = json.dumps(mem, ensure_ascii=False, indent=2)
            return [{"type": "text", "text": f"User memory:\n{body}"}]
        return [{"type": "text", "text": str(mem)}]
    return [{"type": "text", "text": json.dumps(part, ensure_ascii=False)}]


def normalize_messages_for_litellm(messages: list[dict]) -> list[dict]:
    """
    Clients may send non-OpenAI content (e.g. Brave blocks). LiteLLM validates strictly; normalize
    to string or OpenAI multimodal parts before completion.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": "" if content is None else str(content)})
            continue
        new_parts: list[dict] = []
        for block in content:
            if isinstance(block, dict):
                new_parts.extend(_normalize_content_part(block))
            else:
                new_parts.append({"type": "text", "text": str(block)})
        if not new_parts:
            out.append({"role": role, "content": ""})
        elif len(new_parts) == 1 and new_parts[0]["type"] == "text":
            out.append({"role": role, "content": new_parts[0]["text"]})
        elif all(p["type"] == "text" for p in new_parts):
            out.append({"role": role, "content": "\n\n".join(p["text"] for p in new_parts)})
        else:
            out.append({"role": role, "content": new_parts})
    return out


def messages_for_json_storage(messages: list[dict]) -> list[dict]:
    """
    Saved JSON messages: text-only content stays {\"type\": \"text\", \"text\": \"...\"} (legacy).
    If any image_url parts exist, that message's content is a list of OpenAI-style parts with
    full image payloads preserved (deep-copied).
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": {"type": "text", "text": content}})
            continue
        if not isinstance(content, list):
            out.append(
                {
                    "role": role,
                    "content": {"type": "text", "text": "" if content is None else str(content)},
                }
            )
            continue
        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") == "brave-request-summary":
                continue
            for p in _normalize_content_part(block):
                if p.get("type") == "text":
                    t = p.get("text") or ""
                    if t.strip():
                        parts.append({"type": "text", "text": t})
                elif p.get("type") == "image_url":
                    parts.append(copy.deepcopy(p))

        if not parts:
            out.append({"role": role, "content": {"type": "text", "text": ""}})
        elif not any(p.get("type") == "image_url" for p in parts):
            joined = "\n\n".join(p["text"] for p in parts if p.get("type") == "text")
            out.append({"role": role, "content": {"type": "text", "text": joined}})
        else:
            out.append({"role": role, "content": [copy.deepcopy(p) for p in parts]})
    return out


def is_brave_conversation_title_request(request: OpenAIRequest) -> bool:
    """Brave sends a follow-up call with a brave-conversation-title block; skip LLM and disk."""
    for msg in request.messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "brave-conversation-title":
                return True
    return False


def merge_completion_texts(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and str(p).strip())
