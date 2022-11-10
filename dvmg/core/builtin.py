from .base import GeneratorWorkerBase, ObserverBase
from patterns import PatternBase
from typing import Optional


# TODO: #2 Логирование
class GeneratorWorker(GeneratorWorkerBase):

    _observers: list[ObserverBase] = []
    """
    Список подписчиков
    """
    _pattern: PatternBase
    """
    Паттерн расчета модели
    """

    def __init__(self, min_x: float, min_y: float, min_anomaly_height: float,
                 min_end_x: float, max_gap_y_bottom: Optional[float] = None):
        self.min_x = min_x
        self.min_y = min_y
        self.min_anomaly_height = min_anomaly_height
        self.min_end_x = min_end_x
        self.max_gap_y_bottom = max_gap_y_bottom

    def attach(self, observer: ObserverBase) -> None:
        print("GeneratorWorker: Attached an observer %s", observer.__str__())
        self._observers.append(observer)

    def detach(self, observer: ObserverBase) -> None:
        self._observers.remove(observer)

    """
    Методы управления подпиской.
    """

    def set_pattern(self, pattern: PatternBase) -> None:
        self._pattern = pattern

    def notify(self, coordinates: dict[float, float]) -> None:
        """
        Запуск обновления в каждом подписчике.
        """

        for observer in self._observers:
            observer.onNewData(coordinates)

    def run(self, min_x: Optional[float] = None, min_y: Optional[float] = None, min_anomaly_height: Optional[float] = None,
            min_end_x: Optional[float] = None, max_gap_y_bottom: Optional[float] = None) -> None:
        """
        Запуск одной итерации генерации координат
        """
        if min_x is not None and min_y is not None and min_anomaly_height is not None and min_end_x is not None and max_gap_y_bottom is not None:
            self.__init__(min_x, min_y, min_anomaly_height,
                          min_end_x, max_gap_y_bottom)
        self._pattern.random_start_values(
            self.min_x, self.min_y, self.min_anomaly_height, self.min_end_x, self.max_gap_y_bottom)
        coordinates = self._pattern.generate_coordinates()
        self.notify(coordinates)

    def run_mp(self, calls) -> None:
        self.run()
