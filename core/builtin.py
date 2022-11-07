from .base import GeneratorWorkerBase, ObserverBase
from patterns import BasePattern


# TODO: Перевести на multiprocessing
class GeneratorWorker(GeneratorWorkerBase):

    _observers: list[ObserverBase] = []
    """
    Список подписчиков. В реальной жизни список подписчиков может храниться в
    более подробном виде (классифицируется по типу события и т.д.)
    """
    _pattern: BasePattern
    """
    Паттерн расчета модели
    """

    def attach(self, observer: ObserverBase) -> None:
        print("GeneratorWorker: Attached an observer %s", observer.__str__())
        self._observers.append(observer)

    def detach(self, observer: ObserverBase) -> None:
        self._observers.remove(observer)

    """
    Методы управления подпиской.
    """

    def set_pattern(self, pattern: BasePattern) -> None:
        self._pattern = pattern

    def notify(self, coordinates: dict[float, float]) -> None:
        """
        Запуск обновления в каждом подписчике.
        """

        print("GeneratorWorker: Notifying observers...")
        for observer in self._observers:
            observer.onNewData(coordinates)

    def run(self):
        for _ in range(10):
            self._pattern.random_start_values(130, 0.01, 0.1, 0.3)
            coordinates = self._pattern.generate_coordinates()
            self.notify(coordinates)
        print("Done!")
