from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

LITELLM_KNOWN_PREFIXES = (
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


@dataclass
class JudgeConfig:
    """Routing and generation settings for the judge call."""

    provider: Literal["bedrock", "openai"] = "bedrock"
    model: str = ""
    region: str = "us-west-2"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: Optional[float] = None

    def litellm_model(self) -> str:
        m = (self.model or "").strip()
        if not m:
            raise ValueError("JudgeConfig.model is required")
        if self.provider == "bedrock":
            low = m.lower()
            if low.startswith("bedrock/"):
                return m
            return f"bedrock/converse/{m}"
        low = m.lower()
        if any(low.startswith(p) for p in LITELLM_KNOWN_PREFIXES):
            return m
        return f"openai/{m}"

    def completion_kwargs(self) -> dict:
        kw: dict = {}
        if self.provider == "bedrock":
            kw["aws_region_name"] = self.region
        if self.provider == "openai":
            if (self.api_base or "").strip():
                kw["api_base"] = self.api_base.rstrip("/")
            kw["api_key"] = (self.api_key or "").strip() or "dummy"
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return kw
