from .base import CoordinatesProcessorBase

from math import log, pi, sqrt, exp
from typing import Optional
import random


class ExponentialProcessor(CoordinatesProcessorBase):
    """
    Экспоненциальный обработчик лямбды, генерирует события по экспоненциальному закону распределения
    """
    @property
    def to_generate(self) -> int:
        return self.__to_generate

    @to_generate.setter
    def to_generate(self, value: int) -> None:
        self.__to_generate = value

    def __init__(self, to_generate: Optional[int] = None) -> None:
        if to_generate is not None:
            self.__to_generate = to_generate

    def process(self, coordinates: dict[float, float]) -> list[float]:
        """Метод обработки координат, расчитывает ряд случайных событий для экспоненциального закона распределения

        Args:
            coordinates (dict[float, float]): координата x к y(значение от 0 до 1)

        Raises:
            RuntimeError: вызывается в случае, если to_generate не установлен (равен None)

        Returns:
            list[float]: значения x, где есть случайное событие
        """
        if self.__to_generate is None:
            raise RuntimeError(
                "to_generate number not given to ", self.__str__())
        random_numbers: list[float] = [
            random.uniform(0, 1) for _ in range(1, self.__to_generate)]
        # print('I AM HERE@!!!')
        # print(len(random_numbers))
        # print(coordinates)
        processed_numbers: list[float] = [-log(random_numbers[i]) / coordinates[i]
                                          for i in range(len(random_numbers))]
        return processed_numbers


class GaussianProcessor(CoordinatesProcessorBase):
    """
    Гаусовский обработчик лямбды, генерирует события по нормальному закону распределения
    """
    @property
    def to_generate(self) -> int:
        return self.__to_generate

    @to_generate.setter
    def to_generate(self, value: int) -> None:
        self.__to_generate = value

    def __init__(self, to_generate: Optional[int] = None) -> None:
        if to_generate is not None:
            self.__to_generate = to_generate

    def process(self, coordinates: dict[float, float]) -> list[float]:
        """Метод обработки координат, расчитывает ряд случайных событий для нормального закона распределения

        Args:
            coordinates (dict[float, float]): координата x к y(значение от 0 до 1)

        Raises:
            RuntimeError: вызывается в случае, если to_generate не установлен (равен None)

        Returns:
            list[float]: значения x, где есть случайное событие
        """
        if self.__to_generate is None:
            raise RuntimeError(
                "to_generate number not given to ", self.__str__())
        random_numbers: list[float] = [
            random.uniform(0, 1) for _ in range(1, self.__to_generate)]
        # print('I AM HERE@!!!')
        # print(len(random_numbers))
        # print(coordinates)
        processed_numbers: list[float] = [(1 / sqrt(2 * pi)) * exp(-(1 / 2) * (random_numbers[i]**2))
                                          for i in range(len(random_numbers))]
        return processed_numbers
