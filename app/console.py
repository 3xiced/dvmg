"""Файл запуска dvmg с помощью cmd для генерации больших массивов данных"""

from dvmg import worker, patterns, processors
from collections import Counter

import os
import math
import uuid
import inspect


def generate(pattern_name: str, pattern_signature: patterns.PatternBase, _uuid: str, iters: int = 2000, devider: int = 200):
    """Функция генерирует реконструкции фазовых портретов для всех паттернов и записывает их в файлы

    Args:
        pattern_name (str): Название паттерна
        pattern_signature (patterns.PatternBase): Сигнатура паттерна (класс)
        _uuid (str): Хеш код
        iters (int, optional): Кол-во итераций. Defaults to 2000.
        devider (int, optional): Кол-во интервалов. Defaults to 200.

    Raises:
        Exception: Если длина квантилей по x и y не совпадают выбрасывает исключение
    """
    with open("dataset/" + _uuid + f'/{pattern_name}' + ".txt", 'x') as outfile:
        settings = worker.Settings(
            to_generate=1000, min_x=160, min_y=0.01,
            min_anomaly_height=0.2, min_end_x=160,
            x_limit=1000, max_gap_y_bottom=0.2, is_random=True
        )
        generatorWorker = worker.GeneratorWorker(
            settings, pattern_signature(), processors.ExponentialProcessor())  # type: ignore
        print(f'Generating {pattern_signature}')
        for _ in range(iters):
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
            interval_length = max(output_keys) / devider  # CONST
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

            """
            Расчет фазовых портретов
            """
            N = dict(Counter(to_hist))  # Номер интервала - число событий в нем
            # print(N)
            # Коэффициент, какой элемент берем
            # k = 4
            k = 20
            delta_1: list = list()
            delta_k: list = list()
            for i in range(0, devider):  # CONST
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
                if temp_counter == number_of_dots:
                    quantilies_y.append(temp_min_value)
                    temp_min_value = y_coord
                    temp_counter = 0
                temp_counter += 1

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

            """
            Запись в файл
            """
            if len(quantilies_y) != len(quantilies_x):
                raise Exception('Shitass')
            for i in range(len(quantilies_x)):
                outfile.write(str(quantilies_x[i]) +
                              " " + str(quantilies_y[i]) + "\n")


def main():
    """Точка входа в программу
    """
    __patterns_list: list = inspect.getmembers(
        patterns, inspect.isclass)
    __f_patterns_list: list = [
        ptrn for ptrn in __patterns_list if ptrn[0] != 'PatternBase' and ptrn[0] != 'Custom']

    _uuid_ = uuid.uuid4().hex
    if not os.path.isdir('dataset'):
        os.mkdir("dataset")
    os.mkdir("dataset/" + _uuid_)

    for i in range(len(__f_patterns_list)):
        name, signature = __f_patterns_list[i]
        generate(name, signature, _uuid_)


if __name__ == '__main__':
    main()
