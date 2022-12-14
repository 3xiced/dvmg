from collections import Counter
import math


def process_coordinates_to_output(coordinates: list[float]) -> dict:
    """Обрабатывает координаты для последующего вывода на графике либо последующей\n
    обработке координат (e.g. расчет данных для построения гистограммы -> фазовых портретов)

    Args:
        coordinates (list[float]): Список координат после обработки обработчиком

    Returns:
        dict: словарь в формате {x: 0/1, 0 - события нет, 1 - событие есть}
    """
    output: dict = {}
    temp = coordinates[0]
    for i in range(0, len(coordinates)):
        output[temp] = 1
        temp += coordinates[i]
    return output


def process_coordinates_to_histogram(coordinates: list[float], devider: int) -> list[int]:
    """Считает кол-во точек в интервале

    Args:
        coordinates (list[float]): Обработанные координаты
        devider (int): _description_

    Returns:
        list[int]: Число - номер интервала, кол-во чисел - кол-во чисел в этом \n
        интервале (e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])
    """
    to_hist: list = []
    output_keys: list = list.copy(coordinates)
    interval_number = 0
    interval_length = max(output_keys) / devider
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
    return to_hist


def compile_phase_portrait(histogram_data: list[int], devider: int, bias: int) -> tuple[list[int], list[int]]:
    """Генерирует фазовый портрет по данным о кол-ве точек в разных интервалах

    Args:
        histogram_data (list[int]): список из номеров интервалов и кол-ва точек (e.g. [0, 0, 1, 1, 1, 2])
        devider (int): Делитель при разбиении на подинтервалы
        bias (int): Сдвиг по элементам

    Returns:
        tuple[list[int], list[int]]: Кортеж из двух списков, координат по x и по y
    """
    N = dict(Counter(histogram_data))  # Номер интервала - число событий в нем
    # Коэффициент, какой элемент берем
    k = bias
    delta_x: list = list()
    delta_y: list = list()
    for i in range(0, devider):
        if i not in N:
            N[i] = 0

    """
    Формула дельта функций
    """
    for i in range(len(N.keys()) - k):
        """
        Динамические переменные
        """
        # y, x = (N[0] - N[len(N) - 1] - 2 * N[len(N) // 2] - 2 * N[(len(N) // 2) + 1] - 2 * N[(len(N) // 2) - 1] - N[i] - N[i + k],
        #         (N[i + 2 * k] - N[i]) - (N[i + 1] - N[i + k] - (N[0] - N[i] - 2 * N[len(N) - 1])))
        # x, y = ((N[0] - N[i + k // 4]) - ((N[len(N.keys()) - 1] - N[0]) - (
        #     N[len(N.keys()) // 2] - (N[i + k] - N[i] - (N[i + k * 2] - N[i + k // 2])))),
        #     (N[i + 2 * k] - N[i]) - (N[i + 1] - N[i + k] - (N[0] - N[i])))
        x, y = ((N[i] - N[i + 1]),
                (N[i] - N[i + k]))
        delta_x.append(y)
        delta_y.append(x)

    return (delta_x, delta_y)


def compile_phase_reconstruction_quantile(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    """Реконструирует фазовый портрет по методу создания квантилей в двумерном пространстве

    Args:
        delta_x (list[int]): x координаты фазового портрета
        delta_y (list[int]): y координаты фазового портрета

    Returns:
        tuple[list[int], list[int]]: кортеж из координат квантилей по x, y
    """

    """
    Расчет квантилей по x
    """
    # print(delta_1, delta_k)
    _delta_1 = list(x)
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
    Расчет квантилей по y
    """
    _delta_k = list(y)
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

    print(x, y)

    return (quantilies_x, quantilies_y)


def compile_phase_reconstruction_weight_center(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    """Реконструирует фазовый портрет по методу создания квантилей в двумерном пространстве

    Args:
        delta_x (list[int]): x координаты фазового портрета
        delta_y (list[int]): y координаты фазового портрета

    Returns:
        tuple[list[int], list[int]]: кортеж из координат квантилей по x, y
    """

    """
    Расчет квантилей по x
    """
    # print(delta_1, delta_k)
    _delta_1 = list(x)
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
    Расчет квантилей по y
    """
    _delta_k = list(y)
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

    return (quantilies_x, quantilies_y)
