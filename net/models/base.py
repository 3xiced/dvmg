from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union
from tensorflow import Tensor


class ModelBase(ABC):
    """
    Интферфейс модели н.с.
    """
    @abstractmethod
    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        """Метод обучения

        Args:
            x_train (Union[Tensor, list]): Тензор данных для обучения
            y_train (Union[Tensor, list]): Тензор верных ответов для обучения
            batch_size (int, optional): Размер батча. Defaults to 20.
            epochs (int, optional): Число эпох. Defaults to 5.
        """
        ...

    @abstractmethod
    def predict(self, data: Tensor) -> Any:
        """Запуск предсказания

        Args:
            data (Tensor): данные

        Returns:
            Any: Результат работы н.с.
        """
        ...
