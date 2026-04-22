# Browser automation (Leo summarisation)

`[summarise_pages.py](summarise_pages.py)` uses **Playwright** to drive a **Brave** build with Leo AI: for each URL it attaches the open tab, sends a screenshot-based **“Summarise”** request, and relies on Brave talking to your **data API** (`OCELOT_AI_CHAT_SERVER_URL`, default `http://127.0.0.1:8000`). The API writes JSON under `api/output/`; this script does not parse the model reply itself.

Typical entry: **`bash src/data/entrypoint.sh`** from the repo root (see **[`../README.md`](../README.md)**), or run `summarise_pages.py` by hand once the API is up.

## Requirements

- Python deps: **`requirements.txt`** (includes Playwright).
- A one-time **`playwright install`** on the host if browsers are missing (see Playwright docs for your OS).
- A **Brave development** binary; pass **`--browser-executable`** or resolve via **`../resolve_brave_executable.py`**.

## Important: wait time after “Summarise”

After Leo submits the summarisation request, the script **sleeps for a random duration** between **`--min-response-delay`** and **`--max-response-delay`** (seconds). It does **not** wait for a specific DOM signal that the summary finished.

If automation **finishes or closes the tab before the summary is generated** (empty or missing API artifacts, truncated UI, or Leo still streaming), **increase both delays** so the browser stays open longer—for example:

```bash
export SUMMARISE_EXTRA_ARGS="--min-response-delay 45 --max-response-delay 90"
bash src/data/entrypoint.sh --brave-executable /path/to/brave-or-gn-out-dir
```

Or when running `summarise_pages.py` directly:

```bash
python3 src/data/browser_automation/summarise_pages.py \
  --urls-file src/data/urls.txt \
  --browser-executable "$(python3 src/data/resolve_brave_executable.py /path/to/out/Component_arm64)" \
  --min-response-delay 45 \
  --max-response-delay 90
```

Defaults are **20** and **30** seconds. Slow models (e.g. remote Bedrock), heavy pages, or many workers competing for CPU may need **much** higher values.

## Useful CLI flags

| Flag | Default | Notes |
|------|---------|--------|
| `--urls-file` | `urls.txt` | One URL per line. |
| `--num-workers` | `2` | Parallel Brave instances; lower if memory is tight (especially with `--headless`). |
| `--min-response-delay` | `20` | Lower bound of post-submit wait (seconds). |
| `--max-response-delay` | `30` | Upper bound of post-submit wait (seconds). |
| `--browser-executable` | auto | Path to Brave dev binary or use `resolve_brave_executable.py` with GN out dir. |
| `--headless` | (omit) | Omit for a **visible** window (default). Pass **`--headless`** to save memory and run without UI. |
| `--user-data-dir` | temp | Optional persistent profile. |

`--browser-args` can pass extra Chromium flags. **`OCELOT_AI_CHAT_SERVER_URL`** must point at the same host/port Brave uses for the AI chat server (the data API when collecting training JSON).
