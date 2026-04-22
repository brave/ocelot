from __future__ import annotations

from pathlib import Path

import pytest

# test/conftest.py → parent is test/, grandparent is repository root
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
