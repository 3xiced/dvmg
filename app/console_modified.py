"""Файл запуска dvmg с помощью cmd для генерации больших массивов данных"""

from dvmg import worker, patterns, processors
from utils import *
from collections import Counter

import os
import math
import uuid
import inspect
import json

EQUATION_1 = (
    '(N[0] - N[i + k // 4]) - ((N[len(N.keys()) - 1] - N[0]) - (N[len(N.keys()) // 2] - (N[i + k] - N[i] - (N[i + k] - N[i + k // 2]))))',
    '(N[i + k] - N[i]) - (N[i + 1] - N[i + k] - (N[0] - N[i]))')  # sigmoid, reversed_sigmoid, normal - quantile
EQUATION_2 = (
    'N[i]-50*N[i+k]',
    'N[i]-N[0]-N[len(N)-1]-N[k]-N[i+k]'
)  # normal_flipped, plain - quantile

EQUATION_3 = (
    'N[i] - 2*N[len(N)-1] - 4*N[0] - 8*N[1]',
    'N[i]-N[i+k] - N[k] - 2*N[k//2]'
)  # linear_increase, linear_decrease, plain - quantile
EQUATION_4 = (
    'N[i]**N[i+1]-N[i+1]**4-N[len(N)-1]',
    'N[i]+N[k]+N[i+k//2]**N[i+k]'
)  # plain - quantile


def generate(pattern_signature: patterns.PatternBase, devider: int = 200, dx: str | None = None, dy: str | None = None) -> tuple[list[int], list[int]]:
    """Функция генерирует реконструкции фазовых портретов для всех паттернов и записывает их в файлы

    Args:
        pattern_name (str): Название паттерна
        pattern_signature (patterns.PatternBase): Сигнатура паттерна (класс)
        _uuid (str): Хеш код
        iters (int, optional): Кол-во итераций. Defaults to 2000.
        devider (int, optional): Кол-во интервалов. Defaults to 200.

    Returns:
        tuple[list[int], list[int]]: Кортеж квантилей по X and Y

    Raises:
        Exception: Если длина квантилей по x и y не совпадают выбрасывает исключение
    """

    settings = worker.Settings(
        to_generate=1000, min_x=160, min_y=0.01,
        min_anomaly_height=0.2, min_end_x=160,
        x_limit=1000, max_gap_y_bottom=0.2, is_random=True
    ) if pattern_signature != patterns.NormalFlipped else worker.Settings(
        to_generate=1000, min_x=160, min_y=0.01,
        min_anomaly_height=0.2, min_end_x=160,
        x_limit=1000, max_gap_y_bottom=0.79, is_random=True
    )
    generatorWorker = worker.GeneratorWorker(
        settings, pattern_signature(), processors.ExponentialProcessor())  # type: ignore

    _, processed_coordinates = generatorWorker.run()

    output = process_coordinates_to_output(processed_coordinates)

    to_hist = process_coordinates_to_histogram(
        list(output.keys()), devider)

    delta_x, delta_y = compile_phase_portrait(
        to_hist, devider, 30, dx, dy)

    quantilies_x, quantilies_y = compile_phase_reconstruction_quantile(
        delta_x, delta_y)

    if len(quantilies_y) != len(quantilies_x):
        raise Exception('Shitass')

    return (quantilies_x, quantilies_y)


def main():
    """
    Точка входа в программу
    """
    __patterns_list: list = inspect.getmembers(
        patterns, inspect.isclass)
    __f_patterns_list: list = [
        ptrn for ptrn in __patterns_list if ptrn[0] != 'PatternBase' and ptrn[0] != 'Custom']

    _uuid = uuid.uuid4().hex
    if not os.path.isdir('dataset'):
        os.mkdir("dataset")

    data = dict()

    for i in range(len(__f_patterns_list)):
        name, signature = __f_patterns_list[i]
        print(f'Generating {name} - {signature}')
        for _ in range(3000):
            qx1, qy1 = generate(
                signature, dx=EQUATION_1[0], dy=EQUATION_1[1])
            qx2, qy2 = generate(
                signature, dx=EQUATION_2[0], dy=EQUATION_2[1])
            qx3, qy3 = generate(
                signature, dx=EQUATION_3[0], dy=EQUATION_3[1])
            # qx4, qy4 = generate(
            #     signature, dx=EQUATION_4[0], dy=EQUATION_4[1])
            if name not in data:
                data[name] = []
            data[name].append(
                dict(x1=qx1, y1=qy1, x2=qx2, y2=qy2, x3=qx3, y3=qy3))

    json_string = json.dumps(data)
    with open(f'dataset/{_uuid}.json', 'w') as outfile:
        outfile.write(json_string)


if __name__ == '__main__':
    main()
