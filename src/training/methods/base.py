from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import RunConfig


class TrainingMethod(ABC):
    name: str

    @abstractmethod
    def run(self, cfg: RunConfig) -> None: ...


