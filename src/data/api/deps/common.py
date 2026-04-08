"""Minimal common params dependency matching the aichat interface."""

from fastapi import Request

from ..schemas.models import OpenAIRequest


async def create_completion_common_params(
    raw_request: Request,
    request: OpenAIRequest,
) -> dict:
    """Stub returning a dict so the endpoint signature matches."""
    return {
        "model": request.model,
        "is_premium_host": False,
        "has_valid_premium_credential": False,
        "x_forwarded_for": None,
    }
