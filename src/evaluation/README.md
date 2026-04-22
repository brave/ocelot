# LLM-as-judge evaluations for Ocelot

Compare **three** candidate model outputs against the same user prompt using a stronger “judge” model. The judge is invoked through [LiteLLM](https://github.com/BerriAI/litellm), so you can use **AWS Bedrock** or a **local OpenAI-compatible API** (e.g. vLLM).

Layout:

- `judge/` — `JudgeConfig`, prompts/rubric, and `LLMJudge` (`llm_judge.py`)
- `run_judge.py` — CLI over a JSON file of examples
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`

## Install

From the **repository root**:

```bash
pip install -r src/evaluation/requirements.txt
```

Put **`src/`** on `PYTHONPATH` when running modules (or run from an environment that already includes it):

```bash
export PYTHONPATH=/path/to/ocelot/repo/src
```

For Bedrock, configure AWS credentials as usual (`aws configure`, env vars, or an instance role). LiteLLM calls Bedrock via `bedrock/converse/...`.

## Input JSON

A **JSON array** of objects. Each object must have:

- **`messages`**: OpenAI-style chat messages (the prompt every candidate saw).
- **`responses`**: exactly three outputs, either:
  - a JSON object with three keys (model label → response text), or
  - a JSON array of three objects, each with `(name | model | model_name)` and `(text | response | content)`.

Optional fields copied through to the output: **`id`**, **`example_id`**.

## Run with AWS Bedrock

Use `--provider bedrock` (default), your Bedrock model id, and a region:

```bash
python3 -m evaluation.run_judge \
  --input-json path/to/items.json \
  --model bedrock.model:id \
  --region us-west-2 \
  --out path/to/results.json
```


Equivalent environment variables:

- `JUDGE_INPUT_JSON`, `JUDGE_MODEL`, `JUDGE_PROVIDER=bedrock`
- `AWS_DEFAULT_REGION` or `AWS_REGION`

Some models require an **inference profile ARN** instead of a bare model id; pass that string as `--model` if Bedrock returns provisioned-throughput errors.

## Run with local vLLM (OpenAI-compatible)

Point the judge at your server’s base URL (include `/v1`) and the **served model name**:

```bash
python3 -m evaluation.run_judge \
  --input-json path/to/items.json \
  --provider openai \
  --model judge_model \
  --api-base http://127.0.0.1:8000/v1 \
  --out path/to/results.json
```

Useful environment variables:

- `JUDGE_API_BASE` — same as `--api-base`
- `JUDGE_API_KEY` or `OPENAI_API_KEY` — if the gateway requires a key (many local setups accept a dummy value)

## Docker

Build context is the **repository root**; the image sets `PYTHONPATH=/app/src` (see `Dockerfile`).

From the **repository root**:

```bash
docker compose -f src/evaluation/docker-compose.yml run --rm judge \
  --input-json /work/items.json \
  --model YOUR_BEDROCK_OR_VLLM_MODEL_ID \
  --out /work/results.json
```

Mount `src/evaluation/` at `/work` (see `docker-compose.yml`). For vLLM on the host, use `--api-base http://host.docker.internal:8000/v1` (macOS/Windows) or the Docker bridge IP on Linux.

## Programmatic use

With `PYTHONPATH` including **`src/`**:

```python
from evaluation.judge import LLMJudge, JudgeConfig

judge = LLMJudge.for_bedrock("bedrock.model:id", region="us-west-2")
# or
judge = LLMJudge.for_vllm("summariser", "http://127.0.0.1:8000/v1")

sa, sb, sc, reasoning = judge.compare_responses(
    messages,
    [("A", text_a), ("B", text_b), ("C", text_c)],
)
```
