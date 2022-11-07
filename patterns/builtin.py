from .base import BasePattern
import numpy as np


class Sigmoid(BasePattern):
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

    def __init__(self, gap_y_bottom: float = 0.05, gap_y_top: float = 0.8,
                 anomaly_begin_at_x: float = 98, anomaly_width: float = 2,
                 anomaly_height: float = 0.6, min_x: float = 130, min_y: float = 0.01,
                 min_end_x: float = 250) -> None:
        self.__gap_y_bottom = gap_y_bottom
        self.__gap_y_top = gap_y_top - gap_y_bottom
        self.__anomaly_begin_at_x = anomaly_begin_at_x
        self.__anomaly_width = anomaly_width
        self.__anomaly_height = anomaly_height
        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            max_gap_y_bottom: float | None = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise Exception(
                "Maximum bottom gap is bigger than minimum anomaly height.")
        self.__gap_y_bottom = np.random.uniform(
            min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(10, 100)
        self.__anomaly_begin_at_x = np.random.uniform(
            min_x, self.__min_end_x - self.__anomaly_width)

    def function(self, x: float) -> float:
        return 1 / ((1 / self.gap_y_top) + np.exp((-10 / self.anomaly_width) * (x - self.anomaly_begin_at_x - 5))) + self.gap_y_bottom

    def generate_coordinates(self) -> dict[float, float]:
        coordinates: dict[float, float] = dict()
        for x in np.arange(0.0, self.min_end_x, 0.1):
            coordinates[x] = self.function(x)
        return coordinates
