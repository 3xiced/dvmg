from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class GeneratorWorkerBase(ABC):
    """
    Интферфейс генератора объявляет набор методов для управлениями подписчиками.
    """

    @abstractmethod
    def attach(self, observer: WorkerObserverBase) -> None:
        """
        Присоединяет наблюдателя к издателю.
        """
        pass

    @abstractmethod
    def detach(self, observer: WorkerObserverBase) -> None:
        """
        Отсоединяет наблюдателя от издателя.
        """
        pass

    @abstractmethod
    def notify(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        """
        Уведомляет всех наблюдателей о событии.
        """
        pass

    @abstractmethod
    def run(self, min_x: Optional[float] = None, min_y: Optional[float] = None, min_anomaly_height: Optional[float] = None,
            min_end_x: Optional[float] = None, max_gap_y_bottom: Optional[float] = None, x_limit: Optional[int] = None) -> None:
        """
        Запускает расчет координат
        """
        pass

    @abstractmethod
    def run_mp(self, calls) -> None:
        """
        Поддержка многопоточности
        """
        pass


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
        pass
