from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional


class PatternBase(ABC):
    """
    Интферфейс паттерна определяет стандартный набор параметров шаблона и его поведение.
    """

    @abstractproperty
    def gap_y_bottom(self):
        pass

    @abstractproperty
    def gap_y_top(self):
        pass

    @abstractproperty
    def anomaly_begin_at_x(self):
        pass

    @abstractproperty
    def anomaly_width(self):
        pass

    @abstractproperty
    def anomaly_height(self):
        pass

    @abstractproperty
    def min_x(self):
        pass

    @abstractproperty
    def min_y(self):
        pass

    @abstractproperty
    def min_end_x(self):
        pass

    @abstractmethod
    def generate_coordinates(self, x_limit: Optional[int] = None) -> dict[float, float]:
        """
        Генерирует координаты значений лямбды для паттерна.
        """

    @abstractmethod
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float, min_end_x: float, max_gap_y_bottom: float | None = None) -> None:
        """
        Задает случайные gap_y_top, gap_y_bottom, anomaly_begin_at_x, anomaly_width.
        """
        pass

    @abstractmethod
    def function(self, x: int) -> float:
        """
        Описывает математическую функцию паттерна
        """
        pass
