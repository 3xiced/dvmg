from PyQt6 import QtWidgets, QtCore
import sys
import pyqtgraph as pg
import numpy as np
from dvmg import worker, patterns, processors
from ui import Ui_MainWindow
from accessify import private
import inspect
from collections import Counter
import time
pg.setConfigOptions(antialias=True)


class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['adj'] = np.column_stack(
                (np.arange(0, npts - 1), np.arange(1, npts))
            )
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, ev):
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

        ind = self.dragPoint.data()[0]
        if ind != 0 and ind != len(self.data['data']['index']) - 1:
            if ev.pos()[0] + self.dragOffset > self.data['pos'][ind + 1][0] or ev.pos()[0] + self.dragOffset < self.data['pos'][ind - 1][0] or ev.pos()[1] + self.dragOffset <= 0 or ev.pos()[1] + self.dragOffset > 1:
                ev.ignore()
            else:
                self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
                self.data['pos'][ind][0] = ev.pos()[0] + self.dragOffset
        elif ind == 0 and ev.pos()[1] + self.dragOffset > 0 and ev.pos()[1] + self.dragOffset <= 1:
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        elif ind == len(self.data['data']['index']) - 1 and ev.pos()[1] + self.dragOffset > 0 and ev.pos()[1] + self.dragOffset <= 1:
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        self.updateGraph()
        print(ev.pos(), self.dragPoint.data()[
              0], self.data['pos'][ind][1], self.data['pos'][ind][0], ind)
        ev.accept()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Класс формы
    """

    def __init__(self, *args, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.DEVIDER = 20

        self.__graph = Graph()

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
                xMin=1, xMax=1000, yMin=0, yMax=1)
            self.lambdaChartWidget.addItem(self.__graph)
            # Create dots             v - number of dots
            x = np.linspace(1, 1000, 10)
            coordinates = np.column_stack(
                (x, np.array([0.5 for _ in range(len(x))])))
            self.__graph.setData(pos=coordinates, size=10, pxMode=True)
            # Disable sliders
            self.lambdaSlider.setEnabled(False)
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)
        else:
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=1, xMax=100000, yMin=0, yMax=1)
            # Disable sliders
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)

        # Настройка виджета отображения гистограммы
        self.histogramChartWidget.setBackground('transparent')
        self.histogramChartWidget.showGrid(x=True, y=True)
        self.histogramChartWidget.plotItem.vb.setLimits(  # type: ignore
            xMin=1, xMax=self.DEVIDER - 1, yMin=0)

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

    def generate(self) -> None:
        # Create generator instance
        settings = worker.Settings(
            to_generate=self.to_generateSlider.value(), min_x=self.min_xSlider.value(), min_y=self.min_ySlider.value() / 100,
            min_anomaly_height=self.min_anomaly_heightSlider.value() / 100, min_end_x=self.min_xSlider.value(),
            x_limit=self.to_generateSlider.value(), max_gap_y_bottom=self.max_gap_y_bottomSlider.value() / 100, is_random=True
        )
        generatorWorker = worker.GeneratorWorker(
            settings, self.patternsComboBox.currentData()(), self.processorsComboBox.currentData()())

        # Generate coordinates
        coordinates, processed_coordinates = generatorWorker.run()

        # Convert coordinates to output format
        output: dict = {}
        temp = processed_coordinates[0]
        for i in range(0, len(processed_coordinates)):
            output[temp] = 1
            temp += processed_coordinates[i]

        # Histogram
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
        self.lambdaChartWidget.clear()
        self.lambdaChartWidget.plot(
            np.array(list(coordinates.keys())), np.array(list(coordinates.values())), pen=pg.mkPen({'color': "#083ed1"}))

        # Plot histogram
        y, x = np.histogram(to_hist)
        self.histogramChartWidget.clear()
        self.histogramChartWidget.plot(
            x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

    def index_changed(self, _) -> None:
        if self.patternsComboBox.currentText() == 'Custom':
            self.lambdaChartWidget.clear()
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=1, xMax=1000, yMin=0, yMax=1)  # Change limits back to 1000 TODO: 1000 TO CONST
            self.lambdaSlider.setEnabled(False)  # Disable lambda slider

            # Add dots plane
            self.__graph = Graph()
            self.lambdaChartWidget.addItem(self.__graph)
            x = np.linspace(1, 1000, 10)
            coordinates = np.column_stack(
                (x, np.array([0.5 for _ in range(len(x))])))
            self.__graph.setData(pos=coordinates, size=10, pxMode=True)

            # Disable all sliders
            self.min_xSlider.setEnabled(False)
            self.min_anomaly_heightSlider.setEnabled(False)
            self.min_ySlider.setEnabled(False)
            self.to_generateSlider.setEnabled(False)
            self.max_gap_y_bottomSlider.setEnabled(False)
            self.histogramChartWidget.clear()
            self.generatedDataWidget.clear()
            return
        elif self.patternsComboBox.currentText() == 'Plain':
            self.lambdaChartWidget.clear()
            self.lambdaChartWidget.removeItem(
                self.__graph)  # Remove dots plane
            self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
                xMin=1, xMax=100000, yMin=0, yMax=1)  # Change limits to very big number
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
                xMin=1, xMax=100000, yMin=0, yMax=1)  # Change limits to very big number
            self.lambdaChartWidget.removeItem(
                self.__graph)  # Remove dots plane

            # Enable all sliders
            self.min_xSlider.setEnabled(True)
            self.min_anomaly_heightSlider.setEnabled(True)
            self.min_ySlider.setEnabled(True)
            self.to_generateSlider.setEnabled(True)
            self.max_gap_y_bottomSlider.setEnabled(True)
        self.lambdaChartWidget.clear()
        self.histogramChartWidget.clear()
        self.generatedDataWidget.clear()

    @private
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
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec()
