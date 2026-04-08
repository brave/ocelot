from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

from .judge import JudgeConfig, LLMJudge


def _resolve_input_json_path(path: Path) -> Path:
    """Resolve the judge input file; handle common mistaken paths (e.g. /src/evaluation/... in Docker)."""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    pkg_dir = Path(__file__).resolve().parent
    fallback = pkg_dir / path.name
    if fallback.is_file():
        print(
            f"[run_judge] Input not found at {path}; using {fallback}",
            file=sys.stderr,
            flush=True,
        )
        return fallback
    raise SystemExit(
        f"Input file not found: {path}\n"
        "If you use docker compose for this service, the compose file mounts this folder at /work — "
        "pass e.g. --input-json /work/items.json (not /src/evaluation/items.json)."
    )


def _env_float(name: str) -> float | None:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return None
    return float(v)


def _responses_from_record(responses_field: Any) -> List[Tuple[str, str]]:
    if isinstance(responses_field, dict):
        items = list(responses_field.items())
        if len(items) != 3:
            raise ValueError(f"responses object must have exactly 3 keys, got {len(items)}")
        return [(str(k), str(v)) for k, v in items]

    if isinstance(responses_field, list):
        if len(responses_field) != 3:
            raise ValueError(f"responses array must have length 3, got {len(responses_field)}")
        out: List[Tuple[str, str]] = []
        for i, item in enumerate(responses_field):
            if not isinstance(item, dict):
                raise ValueError(f"responses[{i}] must be an object")
            name = item.get("name") or item.get("model") or item.get("model_name")
            text = item.get("text") or item.get("response") or item.get("content")
            if name is None or text is None:
                raise ValueError(
                    f"responses[{i}] needs (name|model|model_name) and (text|response|content)"
                )
            out.append((str(name), str(text)))
        return out

    raise ValueError("responses must be a JSON object or array of 3 objects")


def _judge_from_args(args: argparse.Namespace) -> LLMJudge:
    if args.provider == "bedrock":
        return LLMJudge.for_bedrock(args.model, region=args.region)
    return LLMJudge(
        JudgeConfig(
            provider="openai",
            model=args.model,
            api_base=args.api_base or "",
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )


def main() -> None:
    doc = __doc__ or ""
    desc, _, rest = doc.partition("\n\n")
    p = argparse.ArgumentParser(
        description=desc.strip() or "Run LLM-as-judge over a JSON file of evaluation items.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=rest.strip() or None,
    )
    p.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="JSON file: list of {messages, responses, ...} (default: JUDGE_INPUT_JSON)",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Write JSON array of results (default: stdout)",
    )
    p.add_argument(
        "--provider",
        choices=("bedrock", "openai"),
        default=os.environ.get("JUDGE_PROVIDER", "bedrock").strip().lower(),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", "").strip(),
        help="Bedrock model id or vLLM served model name (default: JUDGE_MODEL)",
    )
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or "us-west-2",
        help="AWS region for Bedrock (default: AWS_DEFAULT_REGION / AWS_REGION / us-west-2)",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("JUDGE_API_BASE", ""),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy",
        help="API key for openai provider (default: JUDGE_API_KEY, OPENAI_API_KEY, or dummy)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=_env_float("JUDGE_TEMPERATURE"),
        help="Optional sampling temperature (default: JUDGE_TEMPERATURE)",
    )

    args = p.parse_args()
    input_path = args.input_json
    if input_path is None:
        env_in = (os.environ.get("JUDGE_INPUT_JSON") or "").strip()
        input_path = Path(env_in) if env_in else None
    if input_path is None:
        p.error("pass --input-json or set JUDGE_INPUT_JSON")
    if not args.model:
        p.error("pass --model or set JUDGE_MODEL")

    if args.provider == "openai" and not (args.api_base or "").strip():
        p.error("--api-base or JUDGE_API_BASE is required when --provider openai")

    input_path = _resolve_input_json_path(input_path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("Input JSON must be a list of objects.", file=sys.stderr)
        raise SystemExit(1)

    print(f"[run_judge] {len(raw)} items, provider={args.provider}, model={args.model}", flush=True)
    judge = _judge_from_args(args)
    results: list[dict] = []

    for idx, record in enumerate(raw):
        if not isinstance(record, dict):
            results.append({"index": idx, "error": "record is not an object"})
            continue
        row_out: dict = {"index": idx}
        for key in ("id", "example_id"):
            if key in record:
                row_out[key] = record[key]
        try:
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError('"messages" must be a JSON array')
            pairs = _responses_from_record(record.get("responses"))
            result = judge.compare_to_result(messages, pairs)
            row_out["scores"] = {
                result.response_names[0]: result.score_a,
                result.response_names[1]: result.score_b,
                result.response_names[2]: result.score_c,
            }
            row_out["winner"] = result.winner
            row_out["judge_reasoning"] = result.judge_reasoning
            row_out["raw_model_output"] = result.raw_model_output
        except Exception as e:
            row_out["error"] = str(e)

        results.append(row_out)

    text = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[run_judge] Wrote {args.out.resolve()}", flush=True)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
