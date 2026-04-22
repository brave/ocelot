"""Tests for data/resolve_brave_executable.resolve."""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import pytest


def _load_resolve_module(root: Path):
    path = root / "src/data/resolve_brave_executable.py"
    spec = importlib.util.spec_from_file_location("resolve_brave_executable", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def resolve_mod(repo_root: Path):
    return _load_resolve_module(repo_root)


def test_resolve_file_returns_resolved_path(resolve_mod, tmp_path: Path) -> None:
    f = tmp_path / "some_binary"
    f.write_bytes(b"")
    out = resolve_mod.resolve(f)
    assert out == f.resolve()


def test_resolve_missing_raises_system_exit(resolve_mod) -> None:
    missing = Path("/nonexistent/path/for/ocelot/test")
    with pytest.raises(SystemExit, match="not found"):
        resolve_mod.resolve(missing)


def test_resolve_darwin_gn_out_dir(resolve_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    inner = (
        tmp_path
        / "Brave Browser Development.app"
        / "Contents"
        / "MacOS"
        / "Brave Browser Development"
    )
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_bytes(b"")
    out = resolve_mod.resolve(tmp_path)
    assert out == inner.resolve()


def test_resolve_linux_gn_out_dir(resolve_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    inner = tmp_path / "brave development"
    inner.write_bytes(b"")
    out = resolve_mod.resolve(tmp_path)
    assert out == inner.resolve()
