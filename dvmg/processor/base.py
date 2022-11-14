from __future__ import annotations
from abc import ABC, abstractmethod


class CoordinatesProcessorBase(ABC):
    """
    Интферфейс обработчика значений лямбды на определенных координатах
    """

    @abstractmethod
    def process(self, coordinates: dict[float, float]) -> list[float]:
        """
        Обрабатывает значения лямбды
        """
        pass
