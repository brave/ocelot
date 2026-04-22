from typing import Any

from pydantic import BaseModel


class OpenAIRequest(BaseModel):
    """Minimal OpenAI chat completion request; accepts extra fields for compatibility."""

    messages: list[dict[str, Any]]
    model: str
    stream: bool | None = False
    max_tokens: int | None = None
    temperature: float | None = None
    tool_choice: str | None = None
    tools: list[dict[str, Any]] | None = None

    class Config:
        extra = "ignore"
