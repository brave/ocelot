# Ocelot data collection

This directory holds the **browser → data API → JSON artifacts** pipeline: Brave visits URLs, the local API records preference-style examples, and postprocessing merges them into train/validation/test splits.

## One-command pipeline

From the **repository root** (use **bash**, not `sh` or `source`):

```bash
bash src/data/entrypoint.sh --brave-executable /path/to/brave-or-gn-out-dir
```

What it does:

1. **`docker compose`** — builds and starts the **data API** (`src/data/docker-compose.yml`), publishes **port 8000**.
2. **Waits** until `http://127.0.0.1:8000/openapi.json` responds.
3. **Virtualenv** — creates `src/data/.venv` if missing and installs `browser_automation/requirements.txt`.
4. **`summarise_pages.py`** — drives Brave against URLs in **`urls.txt`**, using your executable (see below). Requests go to the API on `127.0.0.1:8000` by default.
5. **`merge_leo_outputs.py`** — reads `api/output/*.json`, shuffles, splits **80 / 10 / 10**, writes **`api/output/dataset_split.json`**.
6. **Teardown** — runs **`docker compose down`** so the API container is stopped when the script exits (success or failure).

**Merge only** (no Docker, no browser): recompute the split from existing JSON under `api/output/`:

```bash
bash src/data/entrypoint.sh --merge-only
```

### `--brave-executable`

Pass either:

- The **Mach-O / ELF binary** (e.g. `…/Brave Browser Development.app/Contents/MacOS/Brave Browser Development` on macOS), or  
- The **GN output directory** (e.g. `…/src/out/Component_arm64`).

Resolution is handled by `resolve_brave_executable.py` (same rules as the script docstring).

### Optional environment variables

| Variable | Purpose |
|----------|---------|
| `OCELOT_AI_CHAT_SERVER_URL` | Override API URL for Brave (default `http://127.0.0.1:8000`). |
| `SUMMARISE_NUM_WORKERS` | Parallel Brave workers (default `2`). |
| `SUMMARISE_EXTRA_ARGS` | Extra CLI flags for `summarise_pages.py` (e.g. `--headless`, or longer waits — see below). |

`PYTHONPATH` is set to **`src/`** under the repository root so imports like `evaluation.*` resolve when needed.

**Summarisation still running when the script moves on?** `summarise_pages.py` only waits a random interval between **`--min-response-delay`** and **`--max-response-delay`** after sending “Summarise”; it does not detect completion in the UI. If workers exit too early, raise both values, e.g.  
`SUMMARISE_EXTRA_ARGS="--min-response-delay 45 --max-response-delay 90"`.  
Details: **`browser_automation/README.md`**.

---

## Layout

| Path | Role |
|------|------|
| `entrypoint.sh` | Orchestrates API + venv + summarise + merge + `compose down`. |
| `docker-compose.yml` | API service only (build context: repo root). |
| `urls.txt` | One URL per line for `summarise_pages.py`. |
| `resolve_brave_executable.py` | Resolves GN out dir or binary path to a single executable path. |
| `browser_automation/summarise_pages.py` | Playwright + Brave automation. |
| `browser_automation/README.md` | How it works, CLI flags, **wait-time tuning** if summaries are cut off. |
| `browser_automation/requirements.txt` | Host-side deps for summarise (includes Playwright). |
| `api/` | FastAPI OpenAI-compatible gateway; see **`api/README.md`** for `config/vllm_config.yaml` and behaviour. |
| `api/output/` | Per-request JSON from the API (`*.json`, or paired `*_text.json` / `*_image.json` when multimodal). |
| `postprocessing/merge_leo_outputs.py` | Builds `dataset_split.json` from `api/output`. |
| `filter/` | Optional URL filtering utilities (separate from the entrypoint). |

---

## Running pieces by hand

**API only** (same compose file as the entrypoint):

```bash
docker compose -f src/data/docker-compose.yml up --build
```

Configure LiteLLM / vLLM in `api/config/vllm_config.yaml` (mounted read-only). Details: **`api/README.md`**.

**Summarise only** (API must already be listening on the URL Brave uses):

```bash
export PYTHONPATH=/path/to/ocelot/repo/src
export OCELOT_AI_CHAT_SERVER_URL=http://127.0.0.1:8000
python3 src/data/browser_automation/summarise_pages.py \
  --urls-file src/data/urls.txt \
  --browser-executable "$(python3 src/data/resolve_brave_executable.py /path/to/out/Component_arm64)"
```

**Merge only** (stdlib + script defaults):

```bash
python3 src/data/postprocessing/merge_leo_outputs.py \
  --input-dir src/data/api/output \
  --out src/data/api/output/dataset_split.json
```

---

## Prerequisites

- **Docker** and **Docker Compose** for the API.
- **bash**, **curl**, **Python 3** on the host.
- A **Brave dev/build** binary compatible with your OS (the entrypoint runs automation on the **host**, not inside the API image).
- Playwright may require a one-time **`playwright install`** on the host if imports fail; see Playwright docs for your platform.
