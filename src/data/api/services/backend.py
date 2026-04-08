import asyncio
import os
from typing import Any

import yaml

from ..config.paths import CONFIG_PATH

_LITELLM_KNOWN_PREFIXES = (
    "openai/",
    "azure/",
    "anthropic/",
    "vertex_ai/",
    "gemini/",
    "bedrock/",
    "cohere/",
    "huggingface/",
    "ollama/",
    "deepseek/",
    "mistral/",
    "groq/",
    "together_ai/",
)


def _is_bedrock_arn(s: str) -> bool:
    """True for standard ARNs whose service is bedrock (arn:aws:bedrock:... or GovCloud-style segments)."""
    low = s.strip().lower()
    if not low.startswith("arn:aws:"):
        return False
    parts = low.split(":")
    return len(parts) > 2 and parts[2] == "bedrock"


def _bedrock_litellm_model(s: str) -> str:
    """Prefix raw Bedrock ARNs so LiteLLM gets a provider (avoids model=arn:... without bedrock/)."""
    t = (s or "").strip()
    if not t:
        return t
    low = t.lower()
    if any(low.startswith(p) for p in _LITELLM_KNOWN_PREFIXES):
        return t
    if _is_bedrock_arn(t):
        return f"bedrock/{t}"
    return t


def _ensure_bedrock_converse_route(model: str) -> str:
    """inference ARNs need bedrock/converse/<id>, not bedrock/<id> invoke."""
    t = (model or "").strip()
    if not t:
        return t
    low = t.lower()
    if low.startswith("bedrock/converse/"):
        return t
    if low.startswith("bedrock/"):
        return f"bedrock/converse/{t[8:]}"
    return t


def litellm_model_id(model: str, api_base: str, model_config: dict) -> str:
    """
    Map config `model` to a LiteLLM model string. Bare names (e.g. summariser) are not routable;
    with a custom api_base we default to openai/<model>. Override with litellm_model or litellm_provider.
    Raw Bedrock ARNs become bedrock/converse/<arn> (same route as JudgeConfig for bedrock).
    """
    override = (model_config.get("litellm_model") or "").strip()
    if override:
        out = _bedrock_litellm_model(override)
        return _ensure_bedrock_converse_route(out)
    m = _bedrock_litellm_model((model or "").strip())
    if not m:
        return m
    if not (api_base or "").strip():
        return _ensure_bedrock_converse_route(m)
    low = m.lower()
    if any(low.startswith(p) for p in _LITELLM_KNOWN_PREFIXES):
        return _ensure_bedrock_converse_route(m)
    provider = (model_config.get("litellm_provider") or "openai").strip().lower()
    out = f"{provider}/{m}" if provider else m
    return _ensure_bedrock_converse_route(out)


def load_vllm_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def effective_api_key(full_config: dict, arm_config: dict) -> str:
    """Per-arm api_key overrides top-level; missing arm key inherits from full_config."""
    if "api_key" in arm_config and arm_config["api_key"] is not None:
        return str(arm_config["api_key"]).strip()
    g = full_config.get("api_key")
    return "" if g is None else str(g).strip()


def _bedrock_region_for_litellm(model: str, configured_region: str) -> str:
    """
    Judge always passes aws_region_name (default us-west-2). We must too: wrong/missing region
    can surface as Bedrock HTTP errors; align with evaluation.judge.JudgeConfig.
    """
    r = (configured_region or "").strip()
    if r:
        return r
    low = model.lower()
    marker = "arn:aws:bedrock:"
    i = low.find(marker)
    if i >= 0:
        parts = model[i:].split(":")
        if len(parts) > 3 and parts[3]:
            return parts[3]
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-west-2"
    )


def merge_arm_config(full_config: dict, arm_key: str) -> dict:
    """Copy chosen/rejected block; apply top-level `region` when the arm omits it."""
    arm = dict(full_config.get(arm_key) or {})
    default_region = str(full_config.get("region") or "").strip()
    if default_region and not str(arm.get("region") or "").strip():
        arm = {**arm, "region": default_region}
    return arm


def call_litellm(messages: list[dict], api_base: str, model_config: dict) -> str:
    """Call model via LiteLLM; return assistant content."""
    import litellm

    raw_model = model_config.get("model", "")
    model = litellm_model_id(raw_model, api_base, model_config)
    max_tokens = model_config.get("max_tokens", 1024)
    temperature = model_config.get("temperature", 0.0)
    api_key = (model_config.get("api_key") or "").strip()
    completion_key = api_key if api_key else "EMPTY"
    configured_region = str(model_config.get("region") or "").strip()
    is_bedrock = model.lower().startswith("bedrock/")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if is_bedrock:
        kwargs["aws_region_name"] = _bedrock_region_for_litellm(model, configured_region)
    if is_bedrock:
        # Bedrock uses boto3, not an OpenAI base URL; empty api_base becomes an invalid httpx URL.
        kwargs["drop_params"] = True
    else:
        kwargs["api_base"] = api_base
        kwargs["api_key"] = completion_key
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    return content or ""


async def call_litellm_optional(
    messages: list[dict] | None, api_base: str, model_config: dict
) -> str:
    if not messages:
        return ""
    from .messages import normalize_messages_for_litellm

    normalized = normalize_messages_for_litellm(messages)
    return await asyncio.to_thread(call_litellm, normalized, api_base, model_config)
