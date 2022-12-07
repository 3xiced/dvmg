from .base import GeneratorWorkerBase, WorkerObserverBase, WorkerSettingsBase
from ..processors import CoordinatesProcessorBase
from ..patterns import PatternBase

from typing import Optional


class Settings(WorkerSettingsBase):
    """Класс настроек

        Args:
            to_generate (int): так же x_limit
            min_x (float): _description_
            min_y (float): _description_
            min_anomaly_height (float): _description_
            min_end_x (float): _description_
            x_limit (int): _description_
            is_reversed (Optional[bool], optional): _description_. Defaults to None.
            max_gap_y_bottom (Optional[float], optional): _description_. Defaults to None.
            is_random (Optional[bool], optional): _description_. Defaults to None.
    """

    # region Properties

# Properties for custom generation
    @property
    def gap_y_bottom(self) -> Optional[float]:
        return self.__gap_y_bottom

    @gap_y_bottom.setter
    def gap_y_bottom(self, value: float) -> None:
        self.__gap_y_bottom = value

    @property
    def gap_y_top(self) -> Optional[float]:
        return self.__gap_y_top

    @gap_y_top.setter
    def gap_y_top(self, value: float) -> None:
        self.__gap_y_top = value

    @property
    def anomaly_begin_at_x(self) -> Optional[float]:
        return self.__anomaly_begin_at_x

    @anomaly_begin_at_x.setter
    def anomaly_begin_at_x(self, value: float) -> None:
        self.__anomaly_begin_at_x = value

    @property
    def anomaly_width(self) -> Optional[float]:
        return self.__anomaly_width

    @anomaly_width.setter
    def anomaly_width(self, value: float) -> None:
        self.__anomaly_width = value

    @property
    def anomaly_height(self) -> Optional[float]:
        return self.__anomaly_height

    @anomaly_height.setter
    def anomaly_height(self, value: float) -> None:
        self.__anomaly_height = value

    @property
    def min_x(self) -> float:
        return self.__min_x

    @min_x.setter
    def min_x(self, value: float) -> None:
        self.__min_x = value

    @property
    def min_y(self) -> float:
        return self.__min_y

    @min_y.setter
    def min_y(self, value: float) -> None:
        self.__min_y = value

    @property
    def min_end_x(self) -> float:
        return self.__min_end_x

    @min_end_x.setter
    def min_end_x(self, value: float) -> None:
        self.__min_end_x = value

    @property
    def x_limit(self) -> int:
        return self.__x_limit

    @x_limit.setter
    def x_limit(self, value: int) -> None:
        self.__x_limit = value

    @property
    def is_reversed(self) -> Optional[bool]:
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, value: bool) -> None:
        self.__is_reversed = value

    @property
    def custom_coordinates(self) -> Optional[dict[float, float]]:
        return self.__custom_coordinates

    @custom_coordinates.setter
    def custom_coordinates(self, value: dict[float, float]) -> None:
        self.__custom_coordinates = value

    @property
    def constant_lambda(self) -> Optional[float]:
        return self.__constant_lambda

    @constant_lambda.setter
    def constant_lambda(self, value: float) -> None:
        self.__constant_lambda = value

    # Properties for random generation

    @property
    def to_generate(self) -> int:
        return self.__to_generate

    @to_generate.setter
    def to_generate(self, value: int) -> None:
        self.__to_generate = value

    @property
    def min_anomaly_height(self) -> float:
        return self.__min_anomaly_height

    @min_anomaly_height.setter
    def min_anomaly_height(self, value: float) -> None:
        self.__min_anomaly_height = value

    @property
    def max_gap_y_bottom(self) -> Optional[float]:
        return self.__max_gap_y_bottom

    @max_gap_y_bottom.setter
    def max_gap_y_bottom(self, value: float) -> None:
        self.__max_gap_y_bottom = value

    @property
    def is_random(self) -> Optional[bool]:
        return self.__is_random

    @is_random.setter
    def is_random(self, value: bool) -> None:
        self.__is_random = value

    # endregion

    def __init__(self, to_generate: int, min_x: float,
                 min_y: float, min_anomaly_height: float,
                 min_end_x: float, x_limit: int, is_reversed: Optional[bool] = None,
                 max_gap_y_bottom: Optional[float] = None, is_random: Optional[bool] = None,
                 custom_coordinates: Optional[dict[float, float]] = None,
                 constant_lambda: Optional[float] = None,
                 gap_y_bottom: Optional[float] = None, gap_y_top: Optional[float] = None,
                 anomaly_begin_at_x: Optional[float] = None, anomaly_width: Optional[float] = None,
                 anomaly_height: Optional[float] = None) -> None:
        self.__to_generate = to_generate
        self.__min_x = min_x
        self.__min_y = min_y
        self.__min_anomaly_height = min_anomaly_height
        self.__min_end_x = min_end_x
        self.__x_limit = x_limit
        self.__is_reversed = is_reversed
        self.__max_gap_y_bottom = max_gap_y_bottom
        self.__is_random = is_random
        self.__custom_coordinates = custom_coordinates
        self.__constant_lambda = constant_lambda
        self.__gap_y_bottom = gap_y_bottom
        self.__gap_y_top = gap_y_top
        self.__anomaly_begin_at_x = anomaly_begin_at_x
        self.__anomaly_width = anomaly_width
        self.__anomaly_height = anomaly_height


# TODO: #2 Логирование
class GeneratorWorker(GeneratorWorkerBase):

    _observers: list[WorkerObserverBase] = []
    """
    Список подписчиков
    """

    # region Properties

    @property
    def settings(self) -> WorkerSettingsBase:
        return self.__settings

    @settings.setter
    def settings(self, value: WorkerSettingsBase) -> None:
        self.__settings = value

    @property
    def pattern(self) -> PatternBase:
        return self.__pattern

    @pattern.setter
    def pattern(self, value: PatternBase) -> None:
        self.__pattern = value

    @property
    def coordinates_processor(self) -> CoordinatesProcessorBase:
        return self.__coordinates_processor

    @coordinates_processor.setter
    def coordinates_processor(self, value: CoordinatesProcessorBase) -> None:
        self.__coordinates_processor = value

    # endregion

    def __init__(self, settings: Optional[WorkerSettingsBase] = None,
                 pattern: Optional[PatternBase] = None,
                 coordinates_processor: Optional[CoordinatesProcessorBase] = None) -> None:
        if settings is not None:
            self.__settings = settings
        if pattern is not None:
            self.__pattern = pattern
        if coordinates_processor is not None:
            self.__coordinates_processor = coordinates_processor

    def attach(self, observer: WorkerObserverBase) -> None:
        print("GeneratorWorker: Attached an observer ", observer.__str__())
        self._observers.append(observer)

    def detach(self, observer: WorkerObserverBase) -> None:
        self._observers.remove(observer)

    """
    Методы управления подпиской.
    """

    def notify(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:

        for observer in self._observers:
            observer.onNewData(coordinates, processed_coordinates)

    # TODO: свои исключения
    def run(self, settings: Optional[WorkerSettingsBase] = None) -> tuple[dict[float, float], list[float]]:
        if self.__pattern is None:
            raise RuntimeError("Pattern not given exception.")

        if settings is not None:
            self.__settings = settings

        if self.__settings is None:
            raise RuntimeError("Settings not given.")

        if self.__coordinates_processor is None:
            raise RuntimeError("Coordinates processor not given.")

        self.__pattern.__init__(gap_y_bottom=self.__settings.gap_y_bottom,
                                gap_y_top=self.__settings.gap_y_top,
                                anomaly_begin_at_x=self.__settings.anomaly_begin_at_x,
                                anomaly_width=self.__settings.anomaly_width,
                                anomaly_height=self.__settings.anomaly_height,
                                min_x=self.__settings.min_x,
                                min_y=self.__settings.min_y,
                                min_end_x=self.__settings.min_end_x,
                                x_limit=self.__settings.x_limit,
                                is_reversed=self.__settings.is_reversed,
                                coordinates=self.__settings.custom_coordinates,
                                constant_lambda=self.__settings.constant_lambda)

        if self.__settings.is_random is None or self.__settings.is_random == True:
            self.__pattern.random_start_values(
                self.__settings.min_x, self.__settings.min_y, self.__settings.min_anomaly_height,
                self.__settings.min_end_x, self.__settings.x_limit, self.__settings.max_gap_y_bottom)

        coordinates = self.__pattern.generate_coordinates()

        self.__coordinates_processor.to_generate = self.__settings.to_generate
        processed_coordinates: list[float] = self.__coordinates_processor.process(
            coordinates)
        self.notify(coordinates, processed_coordinates)
        return (coordinates, processed_coordinates)

    def run_mp(self, calls: int) -> None:
        print(f"Processing {calls} call")
        self.run()
