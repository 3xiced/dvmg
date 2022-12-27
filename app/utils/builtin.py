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


def compile_phase_portrait(histogram_data: list[int], devider: int, bias: int, dynamic_equation_x: str | None = None, dynamic_equation_y: str | None = None) -> tuple[list[int], list[int]]:
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
        if dynamic_equation_x is not None and dynamic_equation_y is not None:
            x, y = eval(dynamic_equation_x), eval(dynamic_equation_y)
        else:
            x, y = (N[k] - N[0] - N[len(N) - 1] - N[i],
                    N[i + k] - N[len(N) // 2] - N[0] - 3 * N[len(N) - k])
        delta_x.append(y)
        delta_y.append(x)

    return (delta_x, delta_y)


def compile_phase_reconstruction_quantile(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    """Реконструирует фазовый портрет по методу создания квантилей в двумерном пространстве

    Args:
        delta_x (list[int]): x координаты фазового портрета
        delta_y (list[int]): y координаты фазового портрета

    Returns:
        tuple[list[int], list[int]]: кортеж из координат квантилей по x, y (по 7 точек)
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
    # print(_delta_1)
    for y_coord in _delta_1:
        # print(y_coord)
        # print(temp_counter)
        # print(number_of_dots)
        if temp_counter == number_of_dots:
            quantilies_y.append(temp_min_value)
            temp_min_value = y_coord
            temp_counter = 0
        temp_counter += 1
    # print(quantilies_y)

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
    # print(quantilies_x)

    # print(x, y)

    return (quantilies_x, quantilies_y)


def compile_phase_reconstruction_octante(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    """Реконструирует фазовый портрет по методу создания октантов

    Args:
        delta_x (list[int]): x координаты фазового портрета
        delta_y (list[int]): y координаты фазового портрета

    Returns:
        tuple[list[int], list[int]]: кортеж из координат октантов по x, y (по 8 точек)
    """
    octant_counter: list[int] = [0, 0, 0, 0, 0, 0, 0, 0]

    # print(x, y)
    # print(len(y))
    for i in range(len(y)):
        x_coord = x[i]
        y_coord = y[i]
        # print(x_coord, y_coord)
        # 2 октант
        if x_coord > 0 and y_coord == 0:
            octant_counter[1] += 1
        # 4 октант
        elif x_coord == 0 and y_coord < 0:
            octant_counter[3] += 1
        # 6 октант
        elif x_coord < 0 and y_coord == 0:
            octant_counter[5] += 1
        # 8 октант
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
    octant_coordinates_x: list[int] = [octant_counter[0], octant_counter[1],
                                       octant_counter[2], 0, -octant_counter[4], -octant_counter[5], -octant_counter[6], 0]
    octant_coordinates_y: list[int] = [octant_counter[0], 0, -octant_counter[2], -
                                       octant_counter[3], -octant_counter[4], 0, octant_counter[6], octant_counter[7]]
    # print(octant_coordinates_x, octant_coordinates_y)
    return (octant_coordinates_x, octant_coordinates_y)


def compile_phase_reconstruction_weight_center(x: list[int], y: list[int]) -> tuple[list[float], list[float]]:
    """Реконструирует фазовый портрет по методу выделения центра тяжести по октантам

    Args:
        delta_x (list[int]): x координаты фазового портрета
        delta_y (list[int]): y координаты фазового портрета

    Returns:
        tuple[list[int], list[int]]: кортеж из координат октантов по x, y (по 8 точек)
    """
    octant_counter: list[int] = [0, 0, 0, 0, 0, 0, 0, 0]
    octant_sum: list[list[int]] = [[0, 0], [0, 0], [0, 0],
                                   [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    # print(x, y)
    # print(len(y))
    for i in range(len(y)):
        x_coord = x[i]
        y_coord = y[i]
        # print(x_coord, y_coord)
        # 2 октант
        if x_coord > 0 and y_coord == 0:
            octant_counter[1] += 1
            octant_sum[1][0] += x_coord
            octant_sum[1][1] += y_coord
        # 4 октант
        elif x_coord == 0 and y_coord < 0:
            octant_counter[3] += 1
            octant_sum[3][0] += x_coord
            octant_sum[3][1] += y_coord
        # 6 октант
        elif x_coord < 0 and y_coord == 0:
            octant_sum[5][0] += x_coord
            octant_sum[5][1] += y_coord
            octant_counter[5] += 1
        # 8 октант
        elif x_coord == 0 and y_coord > 0:
            octant_sum[7][0] += x_coord
            octant_sum[7][1] += y_coord
            octant_counter[7] += 1
        # 1 четверть
        elif x_coord > 0 and y_coord > 0:
            # 1 октант
            if y_coord >= x_coord:
                octant_counter[0] += 1
                octant_sum[0][0] += x_coord
                octant_sum[0][1] += y_coord
            # 2 октант
            else:
                octant_counter[1] += 1
                octant_sum[1][0] += x_coord
                octant_sum[1][1] += y_coord
        # 2 четверть
        elif x_coord > 0 and y_coord < 0:
            # 3 отктант
            if y_coord >= -1 * x_coord:
                octant_counter[2] += 1
                octant_sum[2][0] += x_coord
                octant_sum[2][1] += y_coord
            # 4 октант
            else:
                octant_counter[3] += 1
                octant_sum[3][0] += x_coord
                octant_sum[3][1] += y_coord
        # 3 четверть
        elif x_coord < 0 and y_coord < 0:
            # 5 октант
            if y_coord <= x_coord:
                octant_counter[4] += 1
                octant_sum[4][0] += x_coord
                octant_sum[4][1] += y_coord
            # 6 октант
            else:
                octant_counter[5] += 1
                octant_sum[5][0] += x_coord
                octant_sum[5][1] += y_coord
        # 4 четверть
        elif x_coord < 0 and y_coord > 0:
            # 7 октант
            if y_coord <= -1 * x_coord:
                octant_counter[6] += 1
                octant_sum[6][0] += x_coord
                octant_sum[6][1] += y_coord
            # 8 октант
            else:
                octant_counter[7] += 1
                octant_sum[7][0] += x_coord
                octant_sum[7][1] += y_coord
    coordinates_x: list[float] = []
    coordinates_y: list[float] = []

    for i in range(len(octant_counter)):
        coordinates_x.append(octant_sum[i][0] / octant_counter[i]
                             ) if octant_counter[i] != 0 else coordinates_x.append(0)
        coordinates_y.append(octant_sum[i][1] / octant_counter[i]
                             ) if octant_counter[i] != 0 else coordinates_y.append(0)

    # print(coordinates_x, coordinates_y)
    return (coordinates_x, coordinates_y)
