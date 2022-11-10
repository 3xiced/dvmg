from matplotlib import pyplot as plt
from core import *
from patterns import *
from math import log
import multiprocessing
import random
import time
import uuid

TO_GENERATE = 999
BORDER = 10000
DEVIDER = 10
INTERVAL_LENGTH = BORDER / DEVIDER

start = time.perf_counter()

# TODO: #6 построение гистограмм


class Observer(ObserverBase):
    def onNewData(self, coordinates: dict[float, float]) -> None:
        """
        Метод, расчитывающий итоговые данные
        """
        random_numbers: list[float] = [
            random.uniform(0, 1) for _ in range(1, TO_GENERATE)]
        processed_numbers: list[float] = [-log(random_numbers[i]) / coordinates[i]
                                          for i in range(len(random_numbers))]
        """
        Алгоритм генерации случайных событий относительно лямбды (coordinates[i])
        """

        output: dict = {}
        temp = processed_numbers[0]
        for i in range(0, len(processed_numbers)):
            output[temp] = 1
            temp += processed_numbers[i]

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
        plt.savefig(f'images/histograms/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.plot(coordinates.keys(), coordinates.values())
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.savefig(f'images/lambdas/{_uuid}.png')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        plt.close('all')
        plt.rcParams["figure.figsize"] = [20, 15]
        plt.bar(output.keys(), output.values())  # type: ignore
        plt.yticks([1])
        plt.savefig(f'images/events/{_uuid}.png')


generatorWorker = GeneratorWorker(200, 0.01, 0.4, 200, 0.05)
pattern = Sigmoid(min_end_x=1000)
observer = Observer()
generatorWorker.set_pattern(pattern)
generatorWorker.attach(observer)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=5)
    processed_value = pool.map(generatorWorker.run_mp, range(50))
    pool.close()
    pool.join()
    finish = time.perf_counter()
    print(round(finish - start, 2))
