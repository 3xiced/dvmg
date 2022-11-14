from worker import *
from patterns import *
from processor import *

from matplotlib import pyplot as plt
import multiprocessing
import time
import uuid

# XXX: ОТДЕЛЬНЫЙ КОНФИГ
TO_GENERATE = 5000
BORDER = 50000
DEVIDER = 15
INTERVAL_LENGTH = BORDER / DEVIDER


class Renderer(WorkerObserverBase):
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

        for i in range(0, len(output_keys) - 1):
            if output_keys[i] > (max(output_keys) // INTERVAL_LENGTH) * INTERVAL_LENGTH:
                break
            if output_keys[i] > interval_number * INTERVAL_LENGTH and \
                    output_keys[i] <= interval_number * INTERVAL_LENGTH + INTERVAL_LENGTH:
                to_hist += [interval_number]
            else:
                interval_number += 1
                checker = False
                while not checker:
                    if output_keys[i] > interval_number * INTERVAL_LENGTH and \
                            output_keys[i] <= interval_number * INTERVAL_LENGTH + INTERVAL_LENGTH:
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
        plt.ylabel("кол-во событий")
        plt.savefig(f'images/histograms/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.plot(coordinates.keys(), coordinates.values())
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xlabel("время, с")
        plt.ylabel("λ")
        plt.savefig(f'images/lambdas/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.bar(output.keys(), output.values())  # type: ignore
        plt.yticks([1])
        plt.xlabel("время, c")  # Как назвать? Не то
        plt.ylabel("событие")
        plt.savefig(f'images/events/{_uuid}.png')


generatorWorker = GeneratorWorker(1000, 0.01, 0.4, 1000, 0.05, TO_GENERATE)
pattern = Sigmoid(min_end_x=TO_GENERATE)
observer = Renderer()
processor = ExponentialProcessor()
generatorWorker.set_pattern(pattern)
generatorWorker.set_processor(processor)
generatorWorker.attach(observer)

if __name__ == '__main__':
    start = time.perf_counter()
    pool = multiprocessing.Pool(processes=16)
    processed_value = pool.map(generatorWorker.run_mp, range(5))
    pool.close()
    pool.join()
    finish = time.perf_counter()
    print(round(finish - start, 2))
