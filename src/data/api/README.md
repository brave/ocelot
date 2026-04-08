# Ocelot data API

FastAPI service that exposes an OpenAI-style **`POST /v1/chat/completions`** endpoint. For each request it:

1. Builds **training**, **chosen**, and **rejected** message lists (defaults: all match the incoming `messages`).
2. Calls two backends via **LiteLLM** (intended for **vLLM’s OpenAI-compatible** server): one for the chosen completion, one for the rejected completion.
3. Writes a JSON file under **`output/`** with `{ "prompt", "chosen", "rejected" }` (skipped when the body includes a **`brave-conversation-title`** content part — Brave’s follow-up title/bullets call; returns 200 with empty assistant text). Each `prompt` message uses `content: { "type": "text", "text": "..." }` (Brave `brave-request-summary` blocks are omitted). Chosen/rejected are assistant strings.
4. Returns a JSON chat completion whose assistant text is the **chosen** model’s reply.

You must run (or point `api_base` at) an HTTP server that implements the OpenAI chat completions API (e.g. vLLM `--api-key` / `--served-model-name` as you prefer). This compose file only builds the API container.

## Configure `vllm_config.yaml`

Edit `config/vllm_config.yaml` (mounted read-only into the container).

| Field | Meaning |
|--------|--------|
| `api_base` | Base URL for the OpenAI-compatible API (no trailing path beyond what your server expects; LiteLLM uses it with `/chat/completions`). |
| `api_key` | Optional global key; use `""` if the gateway needs no secret. The OpenAI client still requires a value, so an empty config is sent as a harmless placeholder unless you set a real key. Override per arm with `api_key` under `chosen` / `rejected`. |
| `chosen` / `rejected` | Per-arm settings: `model` (served id, e.g. `summariser`), `max_tokens`, `temperature`. With `api_base` set, bare names are sent to LiteLLM as `openai/<model>` so routing works. |
| `litellm_model` | (Optional per arm) Full LiteLLM model id if you do not want the default `openai/<model>` mapping. |
| `litellm_provider` | (Optional per arm) Provider prefix if not `openai` (default). Ignored when `model` already starts with a known provider prefix. |

**Docker networking:** Inside the container, `http://localhost:8000` is this API itself, not your host. To reach vLLM on the host machine:

- **macOS / Windows (Docker Desktop):** e.g. `http://host.docker.internal:<vllm_port>/v1`
- **Linux:** use the host gateway IP or add vLLM as another service in `docker-compose.yml` on the same network and set `api_base` to that service URL.

## Defining Summary prompts

The prompts to generate 'good' and 'bad' summaries are found in `api/services/prompts.py`. Update these prompts to suit the style of summary that is desired.

## Run with Docker Compose

Compose uses build context **`../../..`** (repository root) so the full `src/` tree is available in the image.

```bash
cd src/data/api
docker compose build
docker compose up
```

- **API:** [http://localhost:8000](http://localhost:8000) (e.g. docs at `/docs`).
- **`./output`** on the host is mounted to **`/app/src/data/api/output`** — JSON artifacts appear here.
- **`config/vllm_config.yaml`** is mounted read-only; edit on the host and restart the container to reload.

## Example request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dummy-model-name",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

`model` in the body is echoed in the response; the backends used are those in `vllm_config.yaml` (`chosen` / `rejected`).

With `"stream": true`, the response is **`text/event-stream`** (OpenAI-style SSE): role chunk, one **content** chunk (the chosen summary, or the text `Summary complete.` if empty), a **stop** chunk, then `data: [DONE]`. Work still completes (both models + disk write) before streaming begins.

## Run locally (without Docker)

From the **repository root**, with Python 3.12+ and dependencies installed:

```bash
pip install -r src/data/api/requirements.txt
export PYTHONPATH=/path/to/ocelot/repo/src
uvicorn data.api.main:app --host 0.0.0.0 --port 8000
```

Ensure `config/vllm_config.yaml` and a writable `output/` directory exist under `src/data/api/`.
