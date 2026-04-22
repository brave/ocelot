#!/usr/bin/env python3
"""Print absolute path to Brave dev binary. Arg: executable file or GN out dir (…/out/Component_arm64)."""

import platform
import sys
from pathlib import Path


def resolve(p: Path) -> Path:
    p = p.expanduser()
    if not p.exists():
        raise SystemExit(f"not found: {p}")
    if p.is_file():
        return p.resolve()
    if not p.is_dir():
        raise SystemExit(f"not a file or directory: {p}")

    sysname = platform.system()
    if sysname == "Darwin":
        c = (
            p
            / "Brave Browser Development.app"
            / "Contents"
            / "MacOS"
            / "Brave Browser Development"
        )
    elif sysname == "Linux":
        c = p / "brave development"
    else:
        c = p / "brave development.exe"
    if c.is_file():
        return c.resolve()
    raise SystemExit(
        f"no Brave dev binary under {p.resolve()} "
        "(expected Brave Browser Development.app on macOS, or 'brave development' on Linux)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: resolve_brave_executable.py PATH")
    print(resolve(Path(sys.argv[1])))
