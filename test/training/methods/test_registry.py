from __future__ import annotations

import pytest


def test_get_method_unknown_raises_key_error() -> None:
    from methods.registry import get_method

    with pytest.raises(KeyError, match="Unknown trainer"):
        get_method("not_a_trainer")
