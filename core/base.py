from __future__ import annotations
from abc import ABC, abstractmethod
from patterns import BasePattern


class GeneratorWorkerBase(ABC):
    """
    Интферфейс издателя объявляет набор методов для управлениями подписчиками.
    """

    @abstractmethod
    def attach(self, observer: ObserverBase) -> None:
        """
        Присоединяет наблюдателя к издателю.
        """
        pass

    @abstractmethod
    def detach(self, observer: ObserverBase) -> None:
        """
        Отсоединяет наблюдателя от издателя.
        """
        pass

    @abstractmethod
    def notify(self, coordinates: dict) -> None:
        """
        Уведомляет всех наблюдателей о событии.
        """
        pass

    @abstractmethod
    def set_pattern(self, pattern: BasePattern) -> None:
        """
        Устанавливает паттерн для использования
        """
        pass


class ObserverBase(ABC):
    """
    Интерфейс Наблюдателя объявляет метод уведомления, который издатели
    используют для оповещения своих подписчиков.
    """

    @abstractmethod
    def onNewData(self, coordinates: dict[float, float]) -> None:
        """
        Получить обновление от субъекта.
        """
        pass
