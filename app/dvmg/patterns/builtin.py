from .base import PatternBase

from typing import Optional
import numpy as np

from math import pi, sqrt, exp


class Sigmoid(PatternBase):
    """
    Паттерн сигмоиды
             __________
            /
           /
    ______/
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - 2 * self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        self.__is_reversed = bool(np.random.randint(2))
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)

    # TODO: #5 #4 Возмжность наложения белого шума (дисперсия, мат. ожидание)

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        return 1 / ((1 / self.__gap_y_top) + np.exp((-10 / self.__anomaly_width) *
                                                    (x - self.__anomaly_begin_at_x - self.__anomaly_width / 2))) + self.__gap_y_bottom

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class SigmoidReversed(PatternBase):
    """
    Паттерн обратной сигмоиды
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - 2 * self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        self.__is_reversed = bool(np.random.randint(2))
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)

    # TODO: #5 #4 Возмжность наложения белого шума (дисперсия, мат. ожидание)

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        return 1 / ((1 / self.__gap_y_top) + np.exp((-10 / self.__anomaly_width) * (-1) *
                                                    (x - self.__anomaly_begin_at_x - self.__anomaly_width / 2))) + self.__gap_y_bottom

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class Normal(PatternBase):
    """
    Паттерн Гаусса
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - 2 * self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)
        self.__is_reversed = bool(np.random.randint(2))

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        return self.__gap_y_bottom + ((self.__gap_y_top) * exp((-(x - self.__anomaly_begin_at_x)**2) / (120 * self.__anomaly_width * 0.5**2)))

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class NormalFlipped(PatternBase):
    """
    Паттерн Гаусса перевернутый
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - 2 * self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)
        self.__is_reversed = bool(np.random.randint(2))

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        return 1 + -1 * self.__gap_y_bottom - ((self.__gap_y_top) * exp((-(x - self.__anomaly_begin_at_x)**2) / (120 * self.__anomaly_width * 0.5**2)))

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class LinearIncrease(PatternBase):
    """
    Паттерн линейного возрастания
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        if x <= self.__anomaly_begin_at_x:
            return self.gap_y_bottom
        elif x >= self.__anomaly_begin_at_x + self.__anomaly_width:
            return self.gap_y_top
        else:
            k = (self.gap_y_top - self.gap_y_bottom) / (self.anomaly_width)
            b = self.gap_y_top - k * \
                (self.anomaly_begin_at_x + self.anomaly_width)
            return k * x + b

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class LinearDecrease(PatternBase):
    """
    Паттерн линейного убывания
    """

    # region Properties

    @property
    def gap_y_bottom(self) -> float:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> float:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> float:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> float:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> float:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> bool:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> dict[float, float]:
        ...

    @constant_lambda.setter
    def constant_lambda(self, value: dict[float, float]) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if gap_y_bottom is not None:
            self.__gap_y_bottom = gap_y_bottom
        if gap_y_top is not None and gap_y_bottom is not None:
            self.__gap_y_top = gap_y_top - gap_y_bottom
        if anomaly_begin_at_x is not None:
            self.__anomaly_begin_at_x = anomaly_begin_at_x
        if anomaly_width is not None:
            self.__anomaly_width = anomaly_width
        if anomaly_height is not None:
            self.__anomaly_height = anomaly_height
        if min_x is not None:
            self.__min_x = min_x
        if min_y is not None:
            self.__min_y = min_y
        if min_end_x is not None:
            self.__min_end_x = min_end_x
        if x_limit is not None:
            self.__x_limit = x_limit
        if is_reversed is not None:
            self.__is_reversed = is_reversed

    # TODO: #1 Сделать свои исключения
    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        if max_gap_y_bottom is not None and max_gap_y_bottom > 1 - min_anomaly_height:
            raise RuntimeError(
                "Maximum bottom gap is bigger than minimum anomaly height.")

        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__gap_y_bottom = np.random.uniform(
            self.__min_y, 1 - min_anomaly_height) if max_gap_y_bottom is None else np.random.uniform(
                self.__min_y, max_gap_y_bottom)
        self.__gap_y_top = np.random.uniform(
            self.__gap_y_bottom + min_anomaly_height, 1) - self.__gap_y_bottom
        self.__anomaly_width = np.random.uniform(
            1, self.__x_limit - self.__min_end_x - self.__min_x)
        self.__anomaly_begin_at_x = np.random.uniform(
            self.__min_x, self.__x_limit - self.__min_end_x - self.__anomaly_width)
        # print(self.__min_x, self.__x_limit -
        #       self.__min_end_x - self.__anomaly_width)

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        if x <= self.__anomaly_begin_at_x:
            return self.gap_y_top
        elif x >= self.__anomaly_begin_at_x + self.__anomaly_width:
            return self.gap_y_bottom
        else:
            k = (self.gap_y_top - self.gap_y_bottom) / \
                ((-1) * self.anomaly_width)
            b = self.gap_y_bottom - k * \
                (self.anomaly_begin_at_x + self.anomaly_width)
            return k * x + b

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        # print(self.min_x, self.min_y, self.min_end_x, self.x_limit, self.gap_y_bottom,
        #       self.gap_y_top, self.anomaly_width, self.anomaly_begin_at_x)
        return self.__coordinates


class Plain(PatternBase):
    """
    Паттерн ровной линии лямбда. Как значение лямбды используется параметр gap_y_bottom\n
    ____________________
    """

    # region Properties

    @ property
    def gap_y_bottom(self) -> float:
        ...

    @ gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        ...

    @ property
    def gap_y_top(self) -> float:
        ...

    @ gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        ...

    @ property
    def anomaly_begin_at_x(self) -> float:
        ...

    @ anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        ...

    @ property
    def anomaly_width(self) -> float:
        ...

    @ anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        ...

    @ property
    def anomaly_height(self) -> float:
        ...

    @ anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        ...

    @ property
    def min_x(self) -> float:
        ...

    @ min_x.setter
    def min_x(self, value: float) -> None:
        ...

    @ property
    def min_y(self) -> float:
        ...

    @ min_y.setter
    def min_y(self, value: float) -> None:
        ...

    @ property
    def min_end_x(self) -> float:
        ...

    @ min_end_x.setter
    def min_end_x(self, value: float) -> None:
        ...

    @ property
    def x_limit(self) -> int:
        return self.__x_limit

    @ x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @ property
    def is_reversed(self) -> bool:
        ...

    @ is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        ...

    @ property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @ coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        ...

    @ property
    def constant_lambda(self) -> float:
        return self.__constant_lambda

    @ constant_lambda.setter
    def constant_lambda(self, value: float) -> None:
        self.__constant_lambda = value

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if constant_lambda is not None:
            self.__constant_lambda = constant_lambda
        if x_limit is not None:
            self.__x_limit = x_limit

    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        self.__x_limit = x_limit
        self.__constant_lambda = np.random.uniform(0, 1)

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        return self.__constant_lambda

    def generate_coordinates(self) -> dict[float, float]:
        self.__coordinates: dict[float, float] = dict()
        for x in np.arange(0, self.__x_limit, 1):
            self.__coordinates[x] = self.function(x)
        return self.__coordinates


class Custom(PatternBase):
    """
    Кастомный паттерн, задаются только итоговые координаты, их же и возвращает
    """

    # region Properties

    @ property
    def gap_y_bottom(self) -> float:
        ...

    @ gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        ...

    @ property
    def gap_y_top(self) -> float:
        ...

    @ gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        ...

    @ property
    def anomaly_begin_at_x(self) -> float:
        ...

    @ anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        ...

    @ property
    def anomaly_width(self) -> float:
        ...

    @ anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        ...

    @ property
    def anomaly_height(self) -> float:
        ...

    @ anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        ...

    @ property
    def min_x(self) -> float:
        ...

    @ min_x.setter
    def min_x(self, value: float) -> None:
        ...

    @ property
    def min_y(self) -> float:
        ...

    @ min_y.setter
    def min_y(self, value: float) -> None:
        ...

    @ property
    def min_end_x(self) -> float:
        ...

    @ min_end_x.setter
    def min_end_x(self, value: float) -> None:
        ...

    @ property
    def x_limit(self) -> int:
        ...

    @ x_limit.setter
    def x_limit(self, value: int) -> None:
        ...

    @ property
    def is_reversed(self) -> bool:
        ...

    @ is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        ...

    @ property
    def coordinates(self) -> dict[float, float]:
        return self.__coordinates

    @ coordinates.setter
    def coordinates(self, value: dict[float, float]) -> None:
        self.__coordinates = value

    @ property
    def constant_lambda(self) -> float:
        ...

    @ constant_lambda.setter
    def constant_lambda(self, value: float) -> None:
        ...

    # endregion

    def __init__(self, gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None, min_x: Optional[float] = None,
                 min_y: Optional[float] = None, min_end_x: Optional[float] = None,
                 x_limit: Optional[int] = None, is_reversed: Optional[bool] = None,
                 coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None) -> None:
        if coordinates is not None:
            self.__coordinates = coordinates

    def random_start_values(self, min_x: float, min_y: float, min_anomaly_height: float,
                            min_end_x: float, x_limit: int, max_gap_y_bottom: Optional[float] = None) -> None:
        ...

    def function(self, x: float, dispersion: Optional[float] = None, expected_value: Optional[float] = None) -> float:
        ...

    def generate_coordinates(self) -> dict[float, float]:
        if self.__coordinates is None:
            raise RuntimeError("Coordinates not given to ", self.__str__())
        return self.__coordinates
