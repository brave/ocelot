from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_imports() -> None:
    # Allow imports like `from core.config import RunConfig` when executing from repo root.
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))


def main(argv: list[str] | None = None) -> int:
    _ensure_local_imports()

    from core.config import RunConfig
    from methods.registry import get_method

    cfg = RunConfig.from_argv(argv)
    method = get_method(cfg.trainer)
    method.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


