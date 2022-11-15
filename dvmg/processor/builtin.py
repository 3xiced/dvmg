from .base import CoordinatesProcessorBase
from worker import *
from patterns import *

from math import log
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
        """
        Метод, расчитывающий итоговый ряд по экспоненциальному распределению
        """
        if self.__to_generate is None:
            raise RuntimeError(
                "to_generate number not given to ", self.__str__())
        random_numbers: list[float] = [
            random.uniform(0, 1) for _ in range(1, self.__to_generate)]
        processed_numbers: list[float] = [-log(random_numbers[i]) / coordinates[i]
                                          for i in range(len(random_numbers))]
        return processed_numbers
