from worker import *
from patterns import *
from processor import *

from matplotlib import pyplot as plt
import multiprocessing
import time
import uuid


class Renderer(WorkerObserverBase):

    def __init__(self, devider: int):
        self.devider = devider

    def onNewData(self, coordinates: dict[float, float], processed_coordinates: list[float]) -> None:
        output: dict = {}
        temp = processed_coordinates[0]
        for i in range(0, len(processed_coordinates)):
            output[temp] = 1
            temp += processed_coordinates[i]

        to_hist: list = []
        output_keys: list = list.copy(list(output.keys()))
        interval_number = 0
        """
        Расчет данных для построения графика изменения лямбды
        """
        interval_length = max(output_keys) / self.devider
        for i in range(0, len(output_keys) - 1):
            if output_keys[i] > (max(output_keys) // interval_length) * interval_length:
                break
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
        """
        Расчет данных для построения гистограмм
        """
        _uuid = uuid.uuid4().hex

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.hist(to_hist)
        plt.xlabel("№ интервала")
        plt.ylabel("кол-во событий, N")
        plt.savefig(f'images/histograms/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.plot(coordinates.keys(), coordinates.values())
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xlabel("время, t")
        plt.ylabel("лямбда, λ")
        plt.savefig(f'images/lambdas/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 2]
        plt.bar(output.keys(), output.values())  # type: ignore
        plt.yticks([1])
        plt.xlabel("время, t")
        plt.ylabel("событие")
        plt.savefig(f'images/events/{_uuid}.png')


settings = Settings(
    to_generate=1000, min_x=200, min_y=0.01, min_anomaly_height=0.4,
    min_end_x=200, x_limit=1000, max_gap_y_bottom=0.05)

generatorWorker = GeneratorWorker(Settings(
    to_generate=1000, min_x=200, min_y=0.01, min_anomaly_height=0.4,
    min_end_x=200, x_limit=1000, max_gap_y_bottom=0.05
), Plain(), ExponentialProcessor())

generatorWorker.attach(Renderer(15))

if __name__ == '__main__':
    start = time.perf_counter()
    pool = multiprocessing.Pool(processes=16)
    processed_value = pool.map(generatorWorker.run_mp, range(5))
    pool.close()
    pool.join()
    finish = time.perf_counter()
    print(round(finish - start, 2))
