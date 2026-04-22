from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest


def _tiny_png_data_url() -> str:
    """1×1 PNG as a data URL (matches pipeline `data:image/...;base64,...` decoding)."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (100, 100), color=(220, 20, 60)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def tiny_sft_json(tmp_path: Path) -> Path:
    """Two rows: text-only user message, and user message with a tiny base64 image + text."""
    text_only = {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": "Reply with one word: ok."}]}],
        "chosen": "ok",
        "rejected": "no",
    }
    data_url = _tiny_png_data_url()
    with_image = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What color is the image? Reply with one word."},
                ],
            }
        ],
        "chosen": "Red",
        "rejected": "Blue",
    }
    payload = {"train": [text_only, with_image], "validation": [text_only, with_image]}
    p = tmp_path / "tiny_sft.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture(scope="session", autouse=True)
def _ensure_training_package_on_path(repo_root: Path) -> None:
    d = str((repo_root / "src" / "training").resolve())
    if d not in sys.path:
        sys.path.insert(0, d)
