from PyQt6 import QtWidgets, QtGui, QtCore
import sys  # We need sys so that we can pass argv to QApplication
import pyqtgraph as pg
import numpy as np
sys.path.append("..")
from dvmg import worker, patterns, processors
from MainWindow import Ui_MainWindow
from accessify import private
import inspect

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
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:  # type: ignore
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

    def __init__(self, *args, obj=None, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.__graph = Graph()

        # Заполнение Combo-box
        _translate = QtCore.QCoreApplication.translate
        __patterns_list: list = inspect.getmembers(
            patterns.builtin, inspect.isclass)
        self.__f_patterns_list: list = [
            ptrn for ptrn in __patterns_list if ptrn[0] != 'PatternBase']
        for i in range(len(self.__f_patterns_list)):
            name, signature = self.__f_patterns_list[i]
            self.comboBox.addItem("", userData=signature)
            self.comboBox.setItemText(i, _translate("MainWindow", name))

        # Настройка виджета ручного задавания лямбды
        self.lambdaChartWidget.setBackground('transparent')
        self.lambdaChartWidget.showGrid(x=True, y=True)
        self.lambdaChartWidget.plotItem.vb.setLimits(  # type: ignore
            xMin=1, xMax=1000, yMin=0, yMax=1)
        self.lambdaChartWidget.addItem(self.__graph)

        x = np.linspace(1, 1000, 10)
        coordinates = np.column_stack(
            (x, np.array([0.5 for _ in range(len(x))])))
        self.__graph.setData(pos=coordinates, size=10, pxMode=True)
        if self.comboBox.currentText() != 'Custom':
            self.lambdaChartGroupBox.setEnabled(False)
            self.lambdaChartWidget.removeItem(self.__graph)
        if self.comboBox.currentText() != 'Plain':
            self.lambdaSlider.setEnabled(False)

        # Настройка виджета отображения сгенерированного потока событий
        self.generatedDataWidget.setBackground('transparent')
        self.generatedDataWidget.showGrid(x=True, y=False)
        self.generatedDataWidget.plotItem.vb.setLimits(  # type: ignore
            xMin=0, yMin=0, yMax=1)
        self.generatedDataWidget.getPlotItem().hideAxis('left')  # type: ignore

        # Подключение хендлеров
        self.generateButton.clicked.connect(
            lambda: self.generate())
        self.comboBox.currentIndexChanged.connect(
            self.index_changed)

    def generate(self) -> None:
        # Create generator instance
        print(self.to_generateSlider.value(), self.min_xSlider.value(), self.min_ySlider.value(
        ), self.min_anomaly_heightSlider.value(), self.max_gap_y_bottomSlider.value())
        generatorWorker = worker.GeneratorWorker(worker.Settings(
            to_generate=self.to_generateSlider.value(), min_x=self.min_xSlider.value(), min_y=self.min_ySlider.value() / 100,
            min_anomaly_height=self.min_anomaly_heightSlider.value() / 100, min_end_x=self.min_xSlider.value(),
            x_limit=self.to_generateSlider.value(), max_gap_y_bottom=self.max_gap_y_bottomSlider.value() / 100
        ), self.comboBox.currentData()(), processors.ExponentialProcessor())

        # Generate coordinates
        _, processed_coordinates = generatorWorker.run()

        # Convert coordinates to output format
        output: dict = {}
        temp = processed_coordinates[0]
        for i in range(0, len(processed_coordinates)):
            output[temp] = 1
            temp += processed_coordinates[i]
        self.plot_events(list(output.keys()), list(output.values()))

    def index_changed(self, _) -> None:

        if self.comboBox.currentText() == 'Custom':
            self.lambdaChartGroupBox.setEnabled(True)
            self.lambdaSlider.setEnabled(False)
            self.lambdaChartWidget.addItem(self.__graph)
            return
        elif self.comboBox.currentText() == 'Plain':
            self.lambdaChartGroupBox.setEnabled(False)
            self.lambdaSlider.setEnabled(True)
            self.lambdaChartWidget.removeItem(self.__graph)
            return
        self.lambdaChartGroupBox.setEnabled(False)
        self.lambdaSlider.setEnabled(False)
        self.lambdaChartWidget.removeItem(self.__graph)

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec()


if __name__ == '__main__':
    main()
