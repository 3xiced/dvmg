"""Файл запуска dvmg с помощью cmd для генерации больших массивов данных"""

from dvmg import worker, patterns, processors
from utils import *
from typing import Callable

import os
import inspect
import json

SETTINGS = {
    patterns.Sigmoid: worker.Settings(
        to_generate=1000, min_x=200, min_y=0.05,
        min_anomaly_height=0.6, min_end_x=200,
        x_limit=1000, max_gap_y_bottom=0.1, is_random=True
    ),
    patterns.SigmoidReversed: worker.Settings(
        to_generate=1000, min_x=250, min_y=0.05,
        min_anomaly_height=0.7, min_end_x=250,
        x_limit=1000, max_gap_y_bottom=0.07, is_random=True
    ),
    patterns.Plain: worker.Settings(
        to_generate=1000, min_x=200, min_y=0.1,
        min_anomaly_height=0.89, min_end_x=200,
        x_limit=1000, max_gap_y_bottom=0.1, is_random=True
    ),
    patterns.Normal: worker.Settings(
        to_generate=1000, min_x=180, min_y=0.01,
        min_anomaly_height=0.97, min_end_x=180,
        x_limit=1000, max_gap_y_bottom=0.03, is_random=True
    ),
    patterns.NormalFlipped: worker.Settings(
        to_generate=1000, min_x=400, min_y=0.01,
        min_anomaly_height=0.85, min_end_x=200,
        x_limit=1000, is_random=True
    ),
    patterns.LinearIncrease: worker.Settings(
        to_generate=1000, min_x=100, min_y=0.1,
        min_anomaly_height=0.89, min_end_x=100,
        x_limit=1000, max_gap_y_bottom=0.1, is_random=True
    ),
    patterns.LinearDecrease: worker.Settings(
        to_generate=1000, min_x=100, min_y=0.1,
        min_anomaly_height=0.89, min_end_x=100,
        x_limit=1000, max_gap_y_bottom=0.01, is_random=True
    )
}


def generate(pattern_signature: patterns.PatternBase, reconstruction_method: Callable, bias: int = 20, devider: int = 200, dx: str | None = None, dy: str | None = None) -> tuple[list[int], list[int]]:
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

    settings: worker.Settings = SETTINGS[pattern_signature]  # type: ignore
    generatorWorker = worker.GeneratorWorker(
        settings, pattern_signature(), processors.ExponentialProcessor())  # type: ignore

    _, processed_coordinates = generatorWorker.run()

    output = process_coordinates_to_output(processed_coordinates)

    to_hist = process_coordinates_to_histogram(
        list(output.keys()), devider)

    delta_x, delta_y = compile_phase_portrait(
        to_hist, devider, bias, dx, dy)

    quantilies_x, quantilies_y = reconstruction_method(
        delta_x, delta_y)

    if len(quantilies_y) != len(quantilies_x):
        raise Exception('Shitass')

    return (quantilies_x, quantilies_y)


def main(reconstruction_method: Callable, bias: int = 20, devider: int = 200, dxdy: list[tuple[str, str]] | None = None) -> None:
    """
    Точка входа в программу
    """
    __patterns_list: list = inspect.getmembers(
        patterns, inspect.isclass)
    __f_patterns_list: list = [
        ptrn for ptrn in __patterns_list if ptrn[0] != 'PatternBase' and ptrn[0] != 'Custom']

    # _uuid = uuid.uuid4().hex
    if not os.path.isdir('dataset'):
        os.mkdir("dataset")

    data = dict()
    for i in range(len(__f_patterns_list)):
        name, signature = __f_patterns_list[i]
        print(f'Generating {name} - {signature}')
        for _ in range(3000):
            qxqy_map: dict = dict()
            if dxdy is None:
                qx, qy = generate(signature, reconstruction_method,
                                  bias, devider)
                qxqy_map["x"] = qx
                qxqy_map["y"] = qy
            else:
                for i in range(len(dxdy)):
                    dx, dy = dxdy[i]
                    qx, qy = generate(signature, reconstruction_method,
                                      bias, devider, dx, dy)
                    qxqy_map[f"x{i}"] = qx
                    qxqy_map[f"y{i}"] = qy

            if name not in data:
                data[name] = []
            data[name].append(qxqy_map)

    json_string = json.dumps(data)
    with open(f'dataset/{reconstruction_method.__name__}.json', 'w') as outfile:
        outfile.write(json_string)


if __name__ == '__main__':
    main(compile_phase_reconstruction_quantile)
