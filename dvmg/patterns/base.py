from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class PatternBase(ABC):
    """
    Интферфейс паттерна определяет стандартный набор параметров шаблона и его поведение.
    """

    @property
    def gap_y_bottom(self) -> float:
        ...

    @gap_y_bottom.setter
    @abstractmethod
    def gap_y_bottom(self, value: float) -> None:
        ...

    @property
    def gap_y_top(self) -> float:
        ...

    @gap_y_top.setter
    @abstractmethod
    def gap_y_top(self, value: float) -> None:
        ...

    @property
    def anomaly_begin_at_x(self) -> float:
        ...

    @anomaly_begin_at_x.setter
    @abstractmethod
    def anomaly_begin_at_x(self, value: float) -> None:
        ...

    @property
    def anomaly_width(self) -> float:
        ...

    @anomaly_width.setter
    @abstractmethod
    def anomaly_width(self, value: float) -> None:
        ...

    @property
    def anomaly_height(self) -> float:
        ...

    @anomaly_height.setter
    @abstractmethod
    def anomaly_height(self, value: float) -> None:
        ...

    @property
    def min_x(self) -> float:
        ...

    @min_x.setter
    @abstractmethod
    def min_x(self, value: float) -> None:
        ...

    @property
    def min_y(self) -> float:
        ...

    @min_y.setter
    @abstractmethod
    def min_y(self, value: float) -> None:
        ...

    @property
    def min_end_x(self) -> float:
        ...

    @min_end_x.setter
    @abstractmethod
    def min_end_x(self, value: float) -> None:
        ...

    @property
    def x_limit(self) -> float:
        ...

    @x_limit.setter
    @abstractmethod
    def x_limit(self, value: float) -> None:
        ...

    @property
    def is_reversed(self) -> bool:
        ...

    @is_reversed.setter
    @abstractmethod
    def is_reversed(self, value: bool) -> None:
        ...

    @abstractmethod
    def generate_coordinates(self) -> dict[float, float]:
        """
        Генерирует координаты значений лямбды для паттерна.
        """
        ...

    @abstractmethod
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: float, max_gap_y_bottom: Optional[float]) -> None:
        """
        Задает случайные gap_y_top, gap_y_bottom, anomaly_begin_at_x, anomaly_width.
        """
        ...

    @abstractmethod
    def function(self, x: int) -> float:
        """
        Описывает математическую функцию паттерна
        """
        ...
