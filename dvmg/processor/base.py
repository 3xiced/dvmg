from __future__ import annotations
from abc import ABC, abstractmethod


class CoordinatesProcessorBase(ABC):
    """
    Интферфейс обработчика значений лямбды на определенных координатах
    """

    @property
    def to_generate(self) -> int:
        ...

    @to_generate.setter
    @abstractmethod
    def to_generate(self, value: int) -> None:
        ...

    @abstractmethod
    def process(self, coordinates: dict[float, float]) -> list[float]:
        """Метод обработки координат, расчитывает ряд случайных событий для n-го закона распределения

        Args:
            coordinates (dict[float, float]): координата x к y(значение от 0 до 1)

        Returns:
            list[float]: значения x, где есть случайное событие
        """
        pass
