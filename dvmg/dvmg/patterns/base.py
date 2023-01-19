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

    @property
    def coordinates(self) -> dict[float, float]:
        ...

    @coordinates.setter
    @abstractmethod
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> float:
        ...

    @constant_lambda.setter
    @abstractmethod
    def constant_lambda(self, value: float) -> None:
        ...

    @abstractmethod
    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        """Метод инициализации. Для генерации координат, не прибегая к использованию
        random_start_values, необходимо вручную задать все параметры

        Args:
            gap_y_bottom (Optional[float], optional): расстояние от 0 по y. Defaults to None.
            gap_y_top (Optional[float], optional): расстояние от 1 по y. Defaults to None.
            anomaly_begin_at_x (Optional[float], optional): точка начала аномалии. Defaults to None.
            anomaly_width (Optional[float], optional): ширина аномалии. Defaults to None.
            anomaly_height (Optional[float], optional): высота аномалии. Defaults to None.
            min_x (Optional[float], optional): минимальное расстояние по x. Defaults to None.
            min_y (Optional[float], optional): минимальное расстояние по y. Defaults to None.
            min_end_x (Optional[float], optional): минимальное расстояние по x с конца. Defaults to None.
            x_limit (Optional[int], optional): всего координат. Defaults to None.
            is_reversed (Optional[bool], optional): переворачивает функцию по оси y. Defaults to None.
            coordinates (Optional[dict[float, float]], optional): координаты (используется в Custom pattern). Defaults to None.
            constant_lambda (Optional[float], optional): константа (используется в Plain pattern). Defaults to None.
        """
        ...

    @abstractmethod
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        """Генерирует псевдо-случайные стартовые величины сдвига функции по X, Y, степень ее сжатия и
        ее ориентацию по Y (gap_y_bottom, gap_y_top, anomaly_width, anomaly_begin_at_x, is_reversed)

        Args:
            min_x (float): минимальное расстояние от начала оси координат по X, с которого начнется аномалия
            min_y (float): минимальное расстояние от начала оси координат по Y (минимальное значение λ)
            min_anomaly_height (float): минимальная высота аномалии
            min_end_x (float): минимальное расстояние от конца координат (x_limit) до точки конца аномалии
            x_limit (int): общая длина графика (ограничение по X)
            max_gap_y_bottom (Optional[float]): максимальное значение по Y от начала координат, выше которого
            будет подниматься график (не должно быть меньше, либо равно min_y)
        """
        ...

    @abstractmethod
    def function(self, x: int, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        """Генерирует значение λ для x. При переданной дисперсии и(-ли) мат. ожидания "накладывает" белый шум

        Args:
            x (int): значение x
            dispersion (Optional[float], optional): дисперсия (степень шума). Defaults to None.
            expected_value (Optional[float], optional): мат. ожидание. Defaults to None.

        Returns:
            float: значение функции в точке x
        """
        ...

    @abstractmethod
    def generate_coordinates(self) -> dict[float, float]:
        """Генерирует значения лямбды

        Returns:
            dict[float, float]: координата x к y(значение от 0 до 1)
        """
        ...
