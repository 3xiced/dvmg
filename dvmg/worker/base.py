from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from patterns import PatternBase
from processor import CoordinatesProcessorBase


class GeneratorWorkerBase(ABC):
    """
    Интферфейс генератора объявляет набор методов для управлениями подписчиками.
    """

    @property
    def settings(self) -> WorkerSettingsBase:
        ...

    @settings.setter
    @abstractmethod
    def settings(self, value: WorkerSettingsBase) -> None:
        ...

    @property
    def pattern(self) -> PatternBase:
        ...

    @pattern.setter
    @abstractmethod
    def pattern(self, value: PatternBase) -> None:
        ...

    @property
    def coordinates_processor(self) -> CoordinatesProcessorBase:
        ...

    @coordinates_processor.setter
    @abstractmethod
    def coordinates_processor(self, value: CoordinatesProcessorBase) -> None:
        ...

    @abstractmethod
    def attach(self, observer: WorkerObserverBase) -> None:
        """
        Присоединяет наблюдателя к издателю.
        """
        ...

    @abstractmethod
    def detach(self, observer: WorkerObserverBase) -> None:
        """
        Отсоединяет наблюдателя от издателя.
        """
        ...

    @abstractmethod
    def notify(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        """
        Уведомляет всех наблюдателей о событии.
        """
        ...

    @abstractmethod
    def run(self, min_x: Optional[float] = None, min_y: Optional[float] = None, min_anomaly_height: Optional[float] = None,
            min_end_x: Optional[float] = None, max_gap_y_bottom: Optional[float] = None, x_limit: Optional[int] = None) -> None:
        """
        Запускает расчет координат
        """
        ...

    @abstractmethod
    def run_mp(self, calls) -> None:
        """
        Поддержка многопоточности
        """
        ...


class WorkerObserverBase(ABC):
    """
    Интерфейс Наблюдателя объявляет метод уведомления, который воркер
    использует для оповещения своих подписчиков.
    """

    @abstractmethod
    def onNewData(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        """
        Получает сырые координаты и уже построенные по ним случайные события и делает с ними необходимые операции
        """
        ...


class WorkerSettingsBase(ABC):
    """
    Интерфейс настроек
    """

    @property
    def to_generate(self) -> int:
        ...

    @to_generate.setter
    @abstractmethod
    def to_generate(self, value: int) -> None:
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
    def min_anomaly_height(self) -> float:
        ...

    @min_anomaly_height.setter
    @abstractmethod
    def min_anomaly_height(self, value: float) -> None:
        ...

    @property
    def min_end_x(self) -> float:
        ...

    @min_end_x.setter
    @abstractmethod
    def min_end_x(self, value: float) -> None:
        ...

    @property
    def max_gap_y_bottom(self) -> Optional[float]:
        """НЕОБЯЗАТЕЛЬНО"""  # TODO: убрать этот коментарий на что то по приличнее
        ...

    @max_gap_y_bottom.setter
    @abstractmethod
    def max_gap_y_bottom(self, value: float) -> None:
        ...

    @property
    def x_limit(self) -> float:
        ...

    @x_limit.setter
    @abstractmethod
    def x_limit(self, value: float) -> None:
        ...

    @property
    def is_reversed(self) -> Optional[bool]:
        """НЕОБЯЗАТЕЛЬНО"""  # TODO: убрать этот коментарий на что то по приличнее
        ...

    @is_reversed.setter
    @abstractmethod
    def is_reversed(self, value: bool) -> None:
        ...
