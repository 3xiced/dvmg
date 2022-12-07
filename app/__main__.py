from dvmg import worker, patterns, processors
from PyQt6 import QtWidgets, QtCore
from ui import Ui_MainWindow
from accessify import private, protected
from collections import Counter

import pyqtgraph as pg
import numpy as np
import inspect
import sys
import math


class Graph(pg.GraphItem):
    """Виджет интерактивного графа с точками
    """

    def __init__(self) -> None:
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)

    def setData(self, **kwargs) -> None:
        """Устанавливает стартовые точки
        """
        self.data = kwargs
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['adj'] = np.column_stack(
                (np.arange(0, npts - 1), np.arange(1, npts))
            )
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    @private
    def updateGraph(self) -> None:
        """Апдейтит граф
        """
        pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, ev) -> None:
        """Срабатывает при перетаскивании точки мышью

        Args:
            ev (pyqtgraph.GraphicsScene.mouseEvents.MouseDragEvent): MouseEvent
        """
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)  # type: ignore
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind][1] - pos[1]
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        # Обработка установки новых координат точки. Блокирует перемещение точки дальше следующей и меньше предыдущей
        ind = self.dragPoint.data()[0]
        if ind != 0 and ind != len(self.data['data']['index']) - 1:
            if ev.pos()[1] + self.dragOffset <= 0 or \
                    ev.pos()[1] + self.dragOffset > 1:
                ev.ignore()
            elif ev.pos()[0] + self.dragOffset > self.data['pos'][ind + 1][0] or \
                    ev.pos()[0] + self.dragOffset < self.data['pos'][ind - 1][0]:
                self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
            else:
                self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
                self.data['pos'][ind][0] = ev.pos()[0] + self.dragOffset
        elif (ind == 0 and ev.pos()[1] + self.dragOffset > 0 and ev.pos()[1] + self.dragOffset <= 1) or \
             (ind == len(self.data['data']['index']) - 1 and ev.pos()[1] + self.dragOffset > 0 and ev.pos()[1] + self.dragOffset <= 1):
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset

        self.updateGraph()
        ev.accept()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Класс формы
    """

    @property
    def intervals_devider(self) -> int:
        return self._intervals_devider

    @intervals_devider.setter
    def intervals_devider(self, value: int) -> None:
        self._intervals_devider = value

    @property
    def graph(self) -> Graph:
        return self._graph

    @graph.setter
    def graph(self, value: Graph) -> None:
        self._graph = value

    def __init__(self, *args, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self._intervals_devider = 200
        self._graph = Graph()

        self.DEVIDER = 200

        _translate = QtCore.QCoreApplication.translate

        # Заполнение comboBox (patterns)
        __patterns_list: list = inspect.getmembers(
            patterns, inspect.isclass)
        self.__f_patterns_list: list = [
            ptrn for ptrn in __patterns_list if ptrn[0] != 'PatternBase']
        for i in range(len(self.__f_patterns_list)):
            name, signature = self.__f_patterns_list[i]
            self.patternsComboBox.addItem("", userData=signature)
            self.patternsComboBox.setItemText(
                i, _translate("MainWindow", name))

        # Заполнение comboBox_2 (processors)
        __processors_list: list = inspect.getmembers(
            processors, inspect.isclass)
        self.__f_processors_list: list = [
            prcs for prcs in __processors_list if prcs[0] != 'CoordinatesProcessorBase']
        for i in range(len(self.__f_processors_list)):
            name, signature = self.__f_processors_list[i]
            self.processorsComboBox.addItem("", userData=signature)
            self.processorsComboBox.setItemText(
                i, _translate("MainWindow", name))

        # Настройка виджета ручного задавания лямбды
        self.lambdaChartWidget.setBackground('transparent')
        self.lambdaChartWidget.showGrid(x=True, y=True)
        if self.patternsComboBox.currentText() == 'Custom':
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=0, xMax=1000, yMin=0, yMax=1)
            self.lambdaChartWidget.addItem(self.graph)
            # Create dots             v - number of dots
            x = np.linspace(0, 1000, 11)
            coordinates = np.column_stack(
                (x, np.array([0.5 for _ in range(len(x))])))  # Квази постоянная --
            # (x, np.array([i / 10 for i in range(len(x))]))) # Линейное возврастание /
            # (x, np.array([-i / 10 + 1 for i in range(len(x))])))  # Линейное убывание \
            # (x, np.array([1 / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-(i - 5)**2 / (2 * 1**2)) for i in range(len(x))])))  # Всплеск по гауссу /\
            # (x, np.array([(-1 / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-(i - 5)**2 / (2 * 1**2))) + 1 for i in range(len(x))])))  # Провал по гауссу \/
            # (x, np.array([1 / (1 + np.exp(-2 * i + 10)) for i in range(len(x))])))  # Сигмоида _/
            # (x, np.array([-1 / (1 + np.exp(-2 * i + 10)) + 1 for i in range(len(x))])))  # Обратная сигмоида \_
            self.graph.setData(pos=coordinates, size=11, pxMode=True)
            # Disable sliders
            self.lambdaSlider.setEnabled(False)
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)
        else:
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=0, xMax=100000, yMin=0, yMax=1)
            # Disable sliders
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)

        # Настройка виджета отображения фазового портрета
        self.phaseTraceChartWidget.setBackground('transparent')
        self.phaseTraceChartWidget.showGrid(x=True, y=True)
        # self.histogramChartWidget.plotItem.vb.setLimits(  # type: ignore
        # xMin=1, xMax=self.DEVIDER - 1, yMin=0)

        # Настройка виджета отображения реконструкции фазового портрета
        self.reconstructChartWidget.setBackground('transparent')
        self.reconstructChartWidget.showGrid(x=True, y=True)

        # Настройка виджета отображения сгенерированного потока событий
        self.generatedDataWidget.setBackground('transparent')
        self.generatedDataWidget.showGrid(x=True, y=False)
        self.generatedDataWidget.plotItem.vb.setLimits(  # type: ignore
            xMin=0, yMin=0, yMax=1)
        self.generatedDataWidget.getPlotItem().hideAxis('left')  # type: ignore

        # Подключение хендлеров
        self.generateButton.clicked.connect(
            lambda: self.generate())
        self.patternsComboBox.currentIndexChanged.connect(
            self.index_changed)
        self.customPatternComboBox.currentIndexChanged.connect(
            self.custom_index_changed)

    def generate(self) -> None:
        # Create generator instance
        if self.patternsComboBox.currentText() == 'Custom':
            coords: dict[float, float] = dict()
            for i in range(len(self.graph.data['pos']) - 1):
                left_border = math.ceil(self.graph.data['pos'][i][0])
                right_border = math.floor(self.graph.data['pos'][i + 1][0])

                def func(x: int) -> float: return self.graph.data['pos'][i][1] + (
                    self.graph.data['pos'][i + 1][1] - self.graph.data['pos'][i][1]) /\
                    (self.graph.data['pos'][i + 1][0] - self.graph.data['pos'][i][0]) *\
                    (x - self.graph.data['pos'][i][0])

                for j in range(left_border, right_border + 1):
                    coords[j] = func(j)
            settings = worker.Settings(
                1000, 0, 0, 0, 0, 0, custom_coordinates=coords, is_random=False)  # TODO: 1000 TO CONST
        elif self.patternsComboBox.currentText() == 'Plain':
            settings = worker.Settings(
                1000, 0, 0, 0, 0, x_limit=1000, constant_lambda=self.lambdaSlider.value() / 100, is_random=False)  # TODO: TO CONST
        else:
            settings = worker.Settings(
                to_generate=self.to_generateSlider.value(), min_x=self.min_xSlider.value(), min_y=self.min_ySlider.value() / 100,
                min_anomaly_height=self.min_anomaly_heightSlider.value() / 100, min_end_x=self.min_xSlider.value(),
                x_limit=self.to_generateSlider.value(), max_gap_y_bottom=self.max_gap_y_bottomSlider.value() / 100, is_random=True
            )
        generatorWorker = worker.GeneratorWorker(
            settings, self.patternsComboBox.currentData()(), self.processorsComboBox.currentData()())

        # Generate coordinates
        coordinates, processed_coordinates = generatorWorker.run()

        """
        Обработка координат для вывода на графике
        """
        output: dict = {}
        temp = processed_coordinates[0]
        for i in range(0, len(processed_coordinates)):
            output[temp] = 1
            temp += processed_coordinates[i]

        """
        Расчет данных для гистограммы
        """
        to_hist: list = []
        output_keys: list = list.copy(list(output.keys()))
        interval_number = 0
        interval_length = max(output_keys) / self.DEVIDER
        for i in range(0, len(output_keys) - 1):
            if output_keys[i] > interval_number * interval_length and \
                    output_keys[i] <= interval_number * interval_length + interval_length:
                to_hist += [interval_number]
            else:
                interval_number += 1
                checker = False
                while not checker:
                    if output_keys[i] > interval_number * interval_length and \
                            output_keys[i] <= interval_number * interval_length + interval_length:
                        to_hist += [interval_number]
                        checker = True
                        break
                    interval_number += 1

        # Plot events
        self.plot_events(list(output.keys()), list(output.values()))

        # Plot lambda
        if self.patternsComboBox.currentText() != 'Custom':
            self.lambdaChartWidget.clear()
            self.lambdaChartWidget.plot(
                np.array(list(coordinates.keys())), np.array(list(coordinates.values())), pen=pg.mkPen({'color': "#083ed1"}))

        """
        Расчет фазовых портретов
        """
        N = dict(Counter(to_hist))  # Номер интервала - число событий в нем
        print(N)
        # Коэффициент, какой элемент берем
        # k = 4
        k = 20
        delta_1: list = list()
        delta_k: list = list()
        for i in range(0, self.DEVIDER):
            if i not in N:
                N[i] = 0

        """
        Формула дельта функций
        """
        for i in range(len(N.keys()) - k - k):
            # x, y = ((intervals[0] - intervals[i + k // 4]) - ((intervals[len(intervals.keys()) - 1] - intervals[0]) - (
            #     intervals[len(intervals.keys()) // 2] - (intervals[i + k] - intervals[i] - (intervals[i + k * 2] - intervals[i + k // 2])))),
            #     (intervals[i + 2 * k] - intervals[i]) - (intervals[i + 1] - intervals[i + k] - (intervals[0] - intervals[i])))
            y, x = (N[0] - N[len(N) - 1] - 2 * N[len(N) // 2] - 2 * N[(len(N) // 2) + 1] - 2 * N[(len(N) // 2) - 1] - N[i] - N[i + k],
                    # N[0] - N[len(N) // 2] - N[len(N) - 1] - (N[i + 2 * k] - (N[len(N) - 1] - N[i + k // 2])))
                    (N[i + 2 * k] - N[i]) - (N[i + 1] - N[i + k] - (N[0] - N[i] - 2 * N[len(N) - 1])))
            delta_1.append(y)
            delta_k.append(x)

        octant_counter: list[int] = [0, 0, 0, 0, 0, 0, 0, 0]
        temp_dict: dict[float, float] = dict()
        _delta_1 = list(delta_1)
        for key in delta_k:
            for value in _delta_1:
                temp_dict[key] = value
                _delta_1.remove(value)
                break
        """
        Расчет реконструкции фазовых портретов через разбиение на октанты
        """
        for y_coord in temp_dict.keys():
            x_coord = round(temp_dict[y_coord])
            y_coord = round(y_coord)
            if x_coord > 0 and y_coord == 0:
                octant_counter[1] += 1
            elif x_coord == 0 and y_coord < 0:
                octant_counter[3] += 1
            elif x_coord < 0 and y_coord == 0:
                octant_counter[5] += 1
            elif x_coord == 0 and y_coord > 0:
                octant_counter[7] += 1
            # 1 четверть
            elif x_coord > 0 and y_coord > 0:
                # 1 октант
                if y_coord >= x_coord:
                    octant_counter[0] += 1
                # 2 октант
                else:
                    octant_counter[1] += 1
            # 2 четверть
            elif x_coord > 0 and y_coord < 0:
                # 3 отктант
                if y_coord >= -1 * x_coord:
                    octant_counter[2] += 1
                # 4 октант
                else:
                    octant_counter[3] += 1
            # 3 четверть
            elif x_coord < 0 and y_coord < 0:
                # 5 октант
                if y_coord <= x_coord:
                    octant_counter[4] += 1
                # 6 октант
                else:
                    octant_counter[5] += 1
            # 4 четверть
            elif x_coord < 0 and y_coord > 0:
                # 7 октант
                if y_coord <= -1 * x_coord:
                    octant_counter[6] += 1
                # 8 октант
                else:
                    octant_counter[7] += 1
        octant_coordinates_x: list[int] = [octant_counter[0], octant_counter[1], -
                                           octant_counter[2], 0, -octant_counter[4], -octant_counter[5], -octant_counter[6], 0]
        octant_coordinates_y: list[int] = [octant_counter[0], 0, octant_counter[2], -
                                           octant_counter[3], -octant_counter[4], 0, octant_counter[6], octant_counter[7]]

        """
        Расчет квантилей по y
        """
        # print(delta_1, delta_k)
        _delta_1 = list(delta_1)
        _delta_1.sort()
        quantilies_y: list[int] = list()
        number_of_dots = math.ceil(len(_delta_1) / 8)
        temp_counter = 0
        temp_min_value = min(_delta_1)
        for y_coord in _delta_1:
            # print(temp_counter, number_of_dots, y_coord, len(_delta_1))
            if temp_counter == number_of_dots:
                quantilies_y.append(temp_min_value)
                temp_min_value = y_coord
                temp_counter = 0
            temp_counter += 1
        # print(quantilies_y)

        """
        Расчет квантилей по x
        """
        _delta_k = list(delta_k)
        _delta_k.sort()
        quantilies_x: list[int] = list()
        number_of_dots = math.ceil(len(_delta_k) / 8)
        temp_counter = 0
        temp_min_value = min(_delta_k)
        for x_coord in _delta_k:
            if temp_counter == number_of_dots:
                quantilies_x.append(temp_min_value)
                temp_min_value = x_coord
                temp_counter = 0
            temp_counter += 1
        # print(quantilies_x)

        # delta_1.append(intervals[i + 1] - intervals[i])
        # delta_k.append(intervals[i + k] - intervals[i])
        # print(x, y)

        # Plot histogram
        # y, x = np.histogram(to_hist)
        # self.histogramChartWidget.clear()
        # self.histogramChartWidget.plot(
        #     x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

        # self.phaseTraceChartWidget.clear()
        # self.reconstructChartWidget.clear()
        self.phaseTraceChartWidget.plot(
            delta_k, delta_1, size=10, symbol='o', pen=pg.mkPen({'color': "#083ed190"}))
        # self.reconstructChartWidget.plot(
        #     octant_coordinates_x, octant_coordinates_y, pen=pg.mkPen({'color': "#083ed190"}))
        self.reconstructChartWidget.plot(
            quantilies_x, quantilies_y, symbol='o')

    def index_changed(self, _) -> None:
        if self.patternsComboBox.currentText() == 'Custom':
            self.lambdaChartWidget.clear()
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=0, xMax=1000, yMin=0, yMax=1)  # Change limits back to 1000 TODO: 1000 TO CONST
            self.lambdaSlider.setEnabled(False)  # Disable lambda slider

            # Add dots plane
            self.graph = Graph()
            self.lambdaChartWidget.addItem(self.graph)
            x = np.linspace(0, 1000, 10)
            coordinates = np.column_stack(
                (x, np.array([0.5 for _ in range(len(x))])))
            self.graph.setData(pos=coordinates, size=10, pxMode=True)

            # Disable all sliders
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)
            self.phaseTraceChartWidget.clear()
            self.generatedDataWidget.clear()
            return
        elif self.patternsComboBox.currentText() == 'Plain':
            self.lambdaChartWidget.clear()
            self.lambdaChartWidget.removeItem(
                self.graph)  # Remove dots plane
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=0, xMax=100000, yMin=0, yMax=1)  # Change limits to very big number
            self.lambdaSlider.setEnabled(True)  # Enable lambda slider

            # Disable sliders
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)
        else:
            self.lambdaSlider.setEnabled(False)  # Disable lambda slider
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=0, xMax=100000, yMin=0, yMax=1)  # Change limits to very big number
            self.lambdaChartWidget.removeItem(
                self.graph)  # Remove dots plane

            # Enable all sliders
            self.min_xSlider.setEnabled(True)
            self.min_anomaly_heightSlider.setEnabled(True)
            self.min_ySlider.setEnabled(True)
            self.to_generateSlider.setEnabled(True)
            self.max_gap_y_bottomSlider.setEnabled(True)
        self.lambdaChartWidget.clear()
        self.phaseTraceChartWidget.clear()
        self.generatedDataWidget.clear()

    def custom_index_changed(self, _) -> None:
        self.lambdaChartWidget.removeItem(self.graph)
        self.lambdaChartWidget.addItem(self.graph)
        # Create dots             v - number of dots
        x = np.linspace(0, 1000, 11)
        if self.customPatternComboBox.currentText() == "Постоянная":
            coordinates = np.column_stack(
                (x, np.array([0.5 for _ in range(len(x))])))  # Квази постоянная --
        elif self.customPatternComboBox.currentText() == "Линейное возрастание":
            coordinates = np.column_stack(
                (x, np.array([((i / 10) if i != 0 else 0.01) for i in range(len(x))])))  # Линейное возврастание /
        elif self.customPatternComboBox.currentText() == "Линейное убывание":
            coordinates = np.column_stack(
                (x, np.array([-i / 10 + 1 for i in range(len(x))])))  # Линейное убывание \
        elif self.customPatternComboBox.currentText() == "Всплеск":
            coordinates = np.column_stack(
                (x, np.array([1 / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-(i - 5)**2 / (2 * 1**2)) for i in range(len(x))])))  # Всплеск по гауссу /\
        elif self.customPatternComboBox.currentText() == "Провал":
            coordinates = np.column_stack(
                (x, np.array([(-1 / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-(i - 5)**2 / (2 * 1**2))) + 0.81 for i in range(len(x))])))  # Провал по гауссу \/
        elif self.customPatternComboBox.currentText() == "Сигмоида":
            coordinates = np.column_stack(
                (x, np.array([1 / (1 + np.exp(-2 * i + 10)) for i in range(len(x))])))  # Сигмоида _/
        elif self.customPatternComboBox.currentText() == "Обратная сигмоида":
            coordinates = np.column_stack(
                (x, np.array([-1 / (1 + np.exp(-2 * i + 10)) + 1 for i in range(len(x))])))  # Обратная сигмоида \_\
        else:
            exit()
        self.graph.setData(pos=coordinates, size=11, pxMode=True)

    @ private
    def plot_events(self, x: list[float], y: list[float]) -> None:
        self.generatedDataWidget.clear()
        _y = np.array(y)
        self.generatedDataWidget.plot(
            np.repeat(x, 2),
            np.dstack((np.zeros(_y.shape[0],  # type: ignore
                                dtype=int), _y)).flatten(),
            pen=pg.mkPen({'color': "#083ed1", 'width': 1}),
            connect='pairs',
            name='Stems')


if __name__ == '__main__':
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec()
