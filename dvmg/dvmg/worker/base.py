from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from dvmg.patterns import PatternBase
from dvmg.processors import CoordinatesProcessorBase


class GeneratorWorkerBase(ABC):
    """
    Интферфейс генератора объявляет набор методов для управлениями подписчиками
    и запуском генерирования случайных событий
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
        """Присоединяет наблюдателя к издателю

        Args:
            observer (WorkerObserverBase): наблюдатель
        """
        ...

    @abstractmethod
    def detach(self, observer: WorkerObserverBase) -> None:
        """Отсоединяет наблюдателя

        Args:
            observer (WorkerObserverBase): наблюдатель
        """
        ...

    @abstractmethod
    def notify(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        """Уведомляет всех наблюдателей при событии

        Args:
            coordinates (dict[float, float]): координата x к y(значение от 0 до 1)
            processed_coordinates (list[float]): значения x, где есть случайное событие
        """
        ...

    @abstractmethod
    def run(self, settings: Optional[WorkerSettingsBase] = None) -> tuple[dict[float, float], list[float]]:
        """Запускает расчет координат

        Args:
            min_x (Optional[float], optional): минимальное расстояние от начала оси координат по X, с которого начнется аномалия. Defaults to None.
            min_y (Optional[float], optional): минимальное расстояние от начала оси координат по Y (минимальное значение λ). Defaults to None.
            min_anomaly_height (Optional[float], optional): минимальная высота аномалии. Defaults to None.
            min_end_x (Optional[float], optional): минимальное расстояние от конца координат (x_limit) до точки конца аномалии. Defaults to None.
            max_gap_y_bottom (Optional[float], optional): максимальное значение по Y от начала координат, выше которого
            будет подниматься график (не должно быть меньше, либо равно min_y). Defaults to None.
            x_limit (Optional[int], optional): общая длина графика (ограничение по X). Defaults to None.
        """
        ...

    @abstractmethod
    def run_mp(self, calls: int) -> None:
        """Метод для поддержки многопоточности (multiprocessing)

        Args:
            calls (int): необходимое число итераций генерации
        """
        ...


class WorkerObserverBase(ABC):
    """
    Интерфейс Наблюдателя объявляет метод уведомления, который воркер
    использует для оповещения своих подписчиков.
    """

    @abstractmethod
    def onNewData(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        """Получает сырые координаты и уже построенные по ним случайные события
         и делает с ними любые необходимые операции

        Args:
            coordinates (dict[float, float]): координата x к y(значение от 0 до 1)
            processed_coordinates (list[float]): значения x, где есть случайное событие
        """
        ...


class WorkerSettingsBase(ABC):
    """
    Интерфейс настроек, описывает Properties, которые обязаны быть определены
    у дочернего класса
    """

    # Properties for custom generation
    @property
    def gap_y_bottom(self) -> Optional[float]:
        ...

    @gap_y_bottom.setter
    @abstractmethod
    def gap_y_bottom(self, value: float) -> None:
        ...

    @property
    def gap_y_top(self) -> Optional[float]:
        ...

    @gap_y_top.setter
    @abstractmethod
    def gap_y_top(self, value: float) -> None:
        ...

    @property
    def anomaly_begin_at_x(self) -> Optional[float]:
        ...

    @anomaly_begin_at_x.setter
    @abstractmethod
    def anomaly_begin_at_x(self, value: float) -> None:
        ...

    @property
    def anomaly_width(self) -> Optional[float]:
        ...

    @anomaly_width.setter
    @abstractmethod
    def anomaly_width(self, value: float) -> None:
        ...

    @property
    def anomaly_height(self) -> Optional[float]:
        ...

    @anomaly_height.setter
    @abstractmethod
    def anomaly_height(self, value: float) -> None:
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
    def min_end_x(self) -> float:
        ...

    @min_end_x.setter
    @abstractmethod
    def min_end_x(self, value: float) -> None:
        ...

    @property
    def x_limit(self) -> int:
        ...

    @x_limit.setter
    @abstractmethod
    def x_limit(self, value: int) -> None:
        ...

    @property
    def is_reversed(self) -> Optional[bool]:
        ...

    @is_reversed.setter
    @abstractmethod
    def is_reversed(self, value: bool) -> None:
        ...

    @property
    def custom_coordinates(self) -> Optional[dict[float, float]]:
        ...

    @custom_coordinates.setter
    @abstractmethod
    def custom_coordinates(self, value: dict[float, float]) -> None:
        ...

    @property
    def constant_lambda(self) -> Optional[float]:
        ...

    @constant_lambda.setter
    @abstractmethod
    def constant_lambda(self, value: float) -> None:
        ...

    # Properties for random generation

    @property
    def to_generate(self) -> int:
        ...

    @to_generate.setter
    @abstractmethod
    def to_generate(self, value: int) -> None:
        ...

    @property
    def min_anomaly_height(self) -> float:
        ...

    @min_anomaly_height.setter
    @abstractmethod
    def min_anomaly_height(self, value: float) -> None:
        ...

    @property
    def max_gap_y_bottom(self) -> Optional[float]:
        ...

    @max_gap_y_bottom.setter
    @abstractmethod
    def max_gap_y_bottom(self, value: float) -> None:
        ...

    @property
    def is_random(self) -> Optional[bool]:
        ...

    @is_random.setter
    @abstractmethod
    def is_random(self, value: bool) -> None:
        ...
