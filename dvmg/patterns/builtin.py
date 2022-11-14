from .base import PatternBase

from typing import Optional
import numpy as np


class Sigmoid(PatternBase):
    """
    Паттерн сигмоиды
             __________
            /
           /
    ______/
    """

    @property
    def gap_y_bottom(self):
        return self.__gap_y_bottom

    @property
    def gap_y_top(self):
        return self.__gap_y_top

    @property
    def anomaly_begin_at_x(self):
        return self.__anomaly_begin_at_x

    @property
    def anomaly_width(self):
        return self.__anomaly_width

    @property
    def anomaly_height(self):
        return self.__anomaly_height

    @property
    def min_x(self):
        return self.__min_x

    @property
    def min_y(self):
        return self.__min_y

    @property
    def min_end_x(self):
        return self.__min_end_x

    # TODO: #7 Перегрузить __init__
    def __init__(self, gap_y_bottom: float = 0.05, gap_y_top: float = 0.8,
                 anomaly_begin_at_x: float = 98, anomaly_width: float = 2,
                 anomaly_height: float = 0.6, min_x: float = 130, min_y: float = 0.01,
                 min_end_x: float = 250, x_limit: int = 1000, is_reversed: bool = False) -> None:
        self.__gap_y_bottom = gap_y_bottom
        self.__gap_y_top = gap_y_top - gap_y_bottom
        self.__anomaly_begin_at_x = anomaly_begin_at_x
        self.__anomaly_width = anomaly_width
        self.__anomaly_height = anomaly_height
        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: Optional[int], max_gap_y_bottom: Optional[float] = None) -> None:
        self.__x_limit = x_limit if x_limit is not None else self.__x_limit
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise Exception(
                "Maximum bottom gap is bigger than minimum anomaly height.")
        self.__gap_y_bottom = np.random.uniform(
            min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - 2 * min_end_x - min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            min_x, self.__x_limit - min_end_x - 2 * self.__anomaly_width)
        self.__is_reversed = bool(np.random.randint(2))
        print(min_x, min_end_x, self.__x_limit,
              self.__anomaly_width, self.__anomaly_begin_at_x)

    # TODO: #5 #4 Возмжность наложения белого шума (дисперсия, мат. ожидание)
    def function(self, x: float) -> float:
        return 1 / ((1 / self.gap_y_top) + np.exp((-10 / self.anomaly_width) *
                                                  (-1 if self.__is_reversed else 1) *
                                                  (x - self.anomaly_begin_at_x - self.anomaly_width / 2))) + self.gap_y_bottom

    def generate_coordinates(self, x_limit: Optional[int] = None) -> dict[float, float]:
        if x_limit is not None:
            self.__x_limit = x_limit
        coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            coordinates[x] = self.function(x)
        return coordinates
