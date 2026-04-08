#!/usr/bin/env bash
# Start data API (Docker), set up .venv for browser automation, run summarise + merge, then docker compose down.
#   bash src/data/entrypoint.sh --brave-executable /path/to/out/Component_arm64
#   bash src/data/entrypoint.sh --merge-only
set -euo pipefail

_script="${BASH_SOURCE[0]}"
case "$_script" in /*) ;; *) _script="${PWD}/${_script}" ;; esac
DATA="$(cd "$(dirname "$_script")" && pwd)"
ROOT="$(cd "$DATA/../.." && pwd)"
COMPOSE=(docker compose -f "$DATA/docker-compose.yml")
VENV="$DATA/.venv"
REQ="$DATA/browser_automation/requirements.txt"

merge_only=false
brave=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --merge-only) merge_only=true; shift ;;
    --brave-executable)
      brave="${2:-}"
      [[ -n "$brave" ]] || { echo "error: --brave-executable needs a path" >&2; exit 2; }
      shift 2
      ;;
    *)
      echo "usage: $0 (--brave-executable PATH | --merge-only)" >&2
      exit 2
      ;;
  esac
done

if $merge_only; then
  python3 "$DATA/postprocessing/merge_leo_outputs.py" \
    --input-dir "$DATA/api/output" \
    --out "$DATA/api/output/dataset_split.json"
  exit 0
fi

[[ -n "$brave" ]] || { echo "error: pass --brave-executable or --merge-only" >&2; exit 2; }

cleanup_api() {
  echo "Stopping data API container..." >&2
  "${COMPOSE[@]}" down --remove-orphans || true
}
trap cleanup_api EXIT

"${COMPOSE[@]}" up -d --build

echo "Waiting for http://127.0.0.1:8000 ..."
_el=0
while ! curl -sf "http://127.0.0.1:8000/openapi.json" >/dev/null 2>&1; do
  if [[ "$_el" -ge 120 ]]; then
    echo "timed out waiting for API" >&2
    exit 1
  fi
  sleep 2
  _el=$((_el + 2))
done

browser="$(python3 "$DATA/resolve_brave_executable.py" "$brave")"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$REQ"

export PYTHONPATH="$ROOT/src"
export OCELOT_AI_CHAT_SERVER_URL="${OCELOT_AI_CHAT_SERVER_URL:-http://127.0.0.1:8000}"

"$VENV/bin/python" "$DATA/browser_automation/summarise_pages.py" \
  --urls-file "$DATA/urls.txt" \
  --browser-executable "$browser" \
  --num-workers "${SUMMARISE_NUM_WORKERS:-2}" \
  ${SUMMARISE_EXTRA_ARGS:-}

"$VENV/bin/python" "$DATA/postprocessing/merge_leo_outputs.py" \
  --input-dir "$DATA/api/output" \
  --out "$DATA/api/output/dataset_split.json"
