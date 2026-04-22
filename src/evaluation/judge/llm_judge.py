from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import litellm

from .judge_config import JudgeConfig
from .judge_prompts import build_comparison_messages


@dataclass
class ComparisonResult:
    prompt_messages: list
    response_names: Tuple[str, str, str]
    texts: Tuple[str, str, str]
    score_a: float
    score_b: float
    score_c: float
    judge_reasoning: str
    winner: str
    raw_model_output: str


def _extract_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def parse_judge_output(response_text: str) -> Tuple[Tuple[float, float, float], str]:
    """Parse model output for score_a/b/c and reasoning string."""
    parsed = _extract_json_object(response_text)
    if parsed:
        try:
            sa = float(parsed.get("score_a", 0))
            sb = float(parsed.get("score_b", 0))
            sc = float(parsed.get("score_c", 0))
            reasoning = (
                parsed.get("comparative_analysis")
                or parsed.get("reasoning")
                or response_text
            )
            return (sa, sb, sc), str(reasoning)
        except (TypeError, ValueError):
            pass

    def grab(label: str):
        return re.search(
            rf'["\']?score[_\s]*{label}["\']?[:\s]*(\d+(?:\.\d+)?)',
            response_text,
            re.IGNORECASE,
        )

    ma, mb, mc = grab("a"), grab("b"), grab("c")
    if ma and mb and mc:
        return (
            float(ma.group(1)),
            float(mb.group(1)),
            float(mc.group(1)),
        ), response_text
    raise ValueError(f"Could not parse judge scores from model output: {response_text[:500]!r}...")


class LLMJudge:
    """Judge via LiteLLM (Bedrock converse or OpenAI-compatible / vLLM)."""

    def __init__(
        self,
        config_or_bedrock_model: JudgeConfig | str,
        region: str = "us-west-2",
    ):
        if isinstance(config_or_bedrock_model, JudgeConfig):
            self.config = config_or_bedrock_model
        else:
            self.config = JudgeConfig(
                provider="bedrock",
                model=config_or_bedrock_model,
                region=region,
            )
        litellm.set_verbose = False

    @classmethod
    def for_bedrock(
        cls,
        model: str,
        region: str = "us-west-2",
    ) -> LLMJudge:
        """AWS Bedrock via LiteLLM (`bedrock/converse/...`). Use inference profile ARNs when required."""
        return cls(
            JudgeConfig(
                provider="bedrock",
                model=model,
                region=region,
            )
        )

    @classmethod
    def for_vllm(
        cls,
        model: str,
        api_base: str,
        api_key: Optional[str] = None,
    ) -> LLMJudge:
        """OpenAI-compatible server (e.g. vLLM). `api_base` is typically `http://host:port/v1`."""
        return cls(
            JudgeConfig(
                provider="openai",
                model=model,
                api_base=api_base,
                api_key=api_key,
            )
        )

    def _complete(
        self,
        original_messages: Sequence[dict],
        responses: Sequence[Tuple[str, str]],
    ) -> Tuple[float, float, float, str, str]:
        messages = build_comparison_messages(original_messages, responses)
        model = self.config.litellm_model()
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            **self.config.completion_kwargs(),
        }
        try:
            response = litellm.completion(**kwargs)
            response_text = response.choices[0].message.content or ""
            scores, reasoning = parse_judge_output(response_text)
            return scores[0], scores[1], scores[2], reasoning, response_text
        except ValueError:
            raise
        except Exception as e:
            error_msg = str(e).strip()
            resp = getattr(e, "response", None)
            if resp is not None and hasattr(resp, "text"):
                rt = resp.text
                if callable(rt):
                    try:
                        rt = rt()
                    except Exception:
                        rt = None
                if isinstance(rt, str) and rt.strip():
                    error_msg = rt.strip()
            if not error_msg and hasattr(e, "message"):
                m = getattr(e, "message", None)
                if isinstance(m, str) and m.strip():
                    error_msg = m.strip()
            if not error_msg:
                error_msg = repr(e)

            if self.config.provider == "bedrock" and (
                "provisioned" in error_msg.lower() or "on-demand" in error_msg.lower()
            ):
                error_msg += (
                    "\n\nNote: This model may require provisioned throughput. "
                    "Use an inference profile ARN instead of the model ID. "
                    "Format: arn:aws:bedrock:region:account:inference-profile/profile-name"
                )
            raise RuntimeError(f"Judge completion failed: {error_msg}") from e

    def compare_responses(
        self,
        original_messages: Sequence[dict],
        responses: Sequence[Tuple[str, str]],
    ) -> Tuple[float, float, float, str]:
        """
        Returns (score_a, score_b, score_c, reasoning) aligned with the order of `responses`.
        """
        sa, sb, sc, reasoning, _ = self._complete(original_messages, responses)
        return sa, sb, sc, reasoning

    def compare_to_result(
        self,
        original_messages: Sequence[dict],
        responses: Sequence[Tuple[str, str]],
    ) -> ComparisonResult:
        sa, sb, sc, reasoning, raw = self._complete(original_messages, responses)
        names = tuple(r[0] for r in responses)
        texts = tuple(r[1] for r in responses)
        scores = (sa, sb, sc)
        best_idx = max(range(3), key=lambda i: scores[i])
        winner = names[best_idx]
        return ComparisonResult(
            prompt_messages=list(original_messages),
            response_names=(names[0], names[1], names[2]),
            texts=(texts[0], texts[1], texts[2]),
            score_a=sa,
            score_b=sb,
            score_c=sc,
            judge_reasoning=reasoning,
            winner=winner,
            raw_model_output=raw,
        )


BedrockJudge = LLMJudge
