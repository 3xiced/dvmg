from .base import CoordinatesProcessorBase
from worker import *
from patterns import *

from matplotlib import pyplot as plt
from math import log
import random
import uuid

# TODO: Вынести в отдельный файл с настройками
# XXX: ОТДЕЛЬНЫЙ КОНФИГ
TO_GENERATE = 5000
BORDER = 50000
DEVIDER = 15
INTERVAL_LENGTH = BORDER / DEVIDER


class ExponentialProcessor(CoordinatesProcessorBase):
    """
    Экспоненциальный обработчик лямбды, генерирует события по экспоненциальному закону распределения
    """

    def process(self, coordinates: dict[float, float]) -> list[float]:
        """
        Метод, расчитывающий итоговый ряд по экспоненциальному распределению
        """
        random_numbers: list[float] = [
            random.uniform(0, 1) for _ in range(1, TO_GENERATE)]
        processed_numbers: list[float] = [-log(random_numbers[i]) / coordinates[i]
                                          for i in range(len(random_numbers))]
        return processed_numbers
