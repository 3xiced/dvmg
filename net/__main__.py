# Imports

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../app'))

from os import listdir
from os.path import isfile, join
import app.dvmg.patterns
import app.console as cs
import app.utils as ut
import numpy as np
import json
import inspect
from typing import Callable

import tensorflow as tf
from keras.utils import to_categorical

methods: list[Callable] = [ut.compile_phase_reconstruction_octante,
                           ut.compile_phase_reconstruction_quantile,
                           ut.compile_phase_reconstruction_weight_center]

inputs: dict[Callable, int] = {ut.compile_phase_reconstruction_octante: 14,
                               ut.compile_phase_reconstruction_quantile: 14,
                               ut.compile_phase_reconstruction_weight_center: 16}

method = methods[int(input(
    f"Choose reconstruction method from list {methods} by it's index: 0, 1..."))]

cs.main(method, dx="(N[0] - N[i + k // 4]) - ((N[len(N.keys()) - 1] - N[0]) - (N[len(N.keys()) // 2] - (N[i + k] - N[i] - (N[i + k] - N[i + k // 2]))))",
        dy="(N[i + k] - N[i]) - (N[i + 1] - N[i + k] - (N[0] - N[i]))")

# Get patterns list
patterns_list: list = inspect.getmembers(
    app.dvmg.patterns, inspect.isclass)
f_patterns_list: list = [
    ptrn for ptrn in patterns_list if ptrn[0] != 'PatternBase' and ptrn[0] != 'Custom']

print(f_patterns_list)

# Create DataSet

onlyfiles = [f for f in listdir("./dataset/")
             if isfile(join("./dataset/", f))]

# Create empty dataset lists
dataset_model_1: list[tuple[tuple, str]] = list()  # ((X,y), pattern_name)
# ((test_data_X,test_data_y), pattern_name)
test_dataset_model_1: list[tuple[tuple, str]] = list()

dataset_model_2: list[tuple[np.ndarray, int]] = list()  # (X, class_id)
test_dataset_model_2: list[tuple[np.ndarray, int]] = list()  # (X, class_id)

# Paths to data files
training_file_path = f'dataset/{onlyfiles[0]}'


def parse(path: str, pattern_name: str) -> tuple[np.ndarray, np.ndarray, list]:

    # Данные для обучения для полносвязной НС
    X_model_2: list[tuple[np.ndarray, list]] = list()
    # Данные для обучения (координаты реконструкции фазового портрета) до сортировки для перцептронов
    X_model_1 = np.empty((0, inputs[method]), int)
    # 1 - соответствует правильному паттерну, на который тренируется сеть, 0 - всем остальным до сортировки для перцептронов
    y_model_1 = np.array([])

    # Случайная величина для перемешивания датасета
    random_value = np.array([])
    with open(path, 'r') as json_file:
        data: dict = json.load(json_file)

        for ptrn in list(data.keys()):
            all_coordinates: list = data[ptrn]

            for local_cordinates in all_coordinates:
                X_model_2.append(([np.append(np.array(local_cordinates['x']), np.array(
                    local_cordinates['y'])).tolist()], list(data.keys()).index(ptrn)))  # type: ignore
                X_model_1 = np.append(X_model_1, np.array([np.append(np.array(
                    local_cordinates['x']), np.array(local_cordinates['y'])).tolist()]), axis=0)
                y_model_1 = np.append(
                    y_model_1, 1) if ptrn == pattern_name else np.append(y_model_1, 0)
                # Генерация случайного числа от 0 до 1
                random_value = np.append(random_value, np.random.rand())

    # Сортировка по случайным величинам
    return {  # type: ignore
        "X_model_1": np.array([x for _, x, _ in sorted(zip(random_value, X_model_1, y_model_1), key=lambda x: x[0])]),
        "y_model_1": np.array([y for _, _, y in sorted(zip(random_value, X_model_1, y_model_1), key=lambda x: x[0])]),
        "X_model_2": [x for _, x in sorted(zip(random_value, X_model_2), key=lambda x: x[0])]
    }


for ptrn in f_patterns_list:

    pattern_name, signature = ptrn
    data = parse(training_file_path, pattern_name)

    X_model_1 = data["X_model_1"]  # type: ignore
    y_model_1 = data["y_model_1"]  # type: ignore
    X_model_2 = data["X_model_2"]  # type: ignore

    dataset_model_1 += [((X_model_1, y_model_1), pattern_name)]
    dataset_model_2 = [*dataset_model_2, *X_model_2]

# print(dataset_model_1[0])
# print('\n-----------------\n')
# print(dataset_model_2[0])
# print('\n-----------------\n')

# print(test_dataset_model_1[0])
# print('\n-----------------\n')
# print(test_dataset_model_2[0])
# print('\n-----------------\n')

# print(len(dataset_model_1))

x_train = list()
y_train = list()

for sample in dataset_model_2:
    x_train.append(sample[0][0])
    y_train.append(sample[1])

x_train = tf.convert_to_tensor(x_train)
y_train = to_categorical(y_train, 7)

print(x_train, y_train)

import models

models_list: list = inspect.getmembers(
    models, inspect.isclass)

model: models.ModelBase = models_list[int(input(
    f"Choose model to use from models list {models_list} by its index 0, 1..."))][1]()  # type: ignore
model.fit(x_train, y_train, epochs=10)  # type: ignore

# Test on real data
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from app.utils import *

lines: list = open('./net/data/cameraDetections.txt').read().splitlines()

timestamps: list = list()

for line in lines:
    if '2021-' in line:
        continue
    if '+04' in line:
        line = line.replace('+04', '')
    if '.' not in line:
        line += '.0'
    time = datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f') - timedelta(hours=1)
    timestamps.append(time.timestamp())

dt_strings: list[str] = list()
dt_objects: list[datetime] = list()
for tmsp in timestamps:
    dt_strings += [datetime.fromtimestamp(
        tmsp).strftime('%Y-%m-%d %H:%M:%S.%f')]
    dt_objects += [datetime.fromtimestamp(tmsp)]

DEVIDER = 100

counter = 0
start_time = dt_objects[0] - timedelta(minutes=15)
end_time = start_time + timedelta(hours=1)
while end_time < dt_objects[-1]:
    counter += 1
    start_time = start_time + timedelta(minutes=15)
    end_time = start_time + timedelta(hours=1)

    to_inspect_timestamps: list[float] = [
        dt.timestamp() for dt in dt_objects if dt > start_time and dt <= end_time]
    to_inspect_timestamps_dt: list[str] = [dt.strftime(
        '%Y-%m-%d %H:%M:%S.%f') for dt in dt_objects if dt > start_time and dt <= end_time]
    # print(to_inspect_timestamps_dt)

    output: dict = dict()
    cut_timestamp = [to_inspect_timestamps[i] -
                     min(to_inspect_timestamps) + .001 for i in range(len(to_inspect_timestamps))]
    if len(cut_timestamp) == 0:
        continue

    for time in cut_timestamp:
        output[time] = 1
    # print(list(output.keys()))
    to_hist = process_coordinates_to_histogram(list(output.keys()), DEVIDER)
    x, y = compile_phase_portrait(to_hist, DEVIDER, 20)
    rx, ry = method(x, y)
    test_data = tf.convert_to_tensor(np.array(rx + ry), tf.float32)

    # Predict and print
    prediction = model.predict(test_data)  # type: ignore
    tt = start_time.strftime('%Y-%m-%d %H:%M:%S').replace(':', '-')
    class_name = f_patterns_list[prediction.tolist()[0].index(
        max(prediction.tolist()[0]))][0]
    print(f"ITER [{counter}] TIME: {tt} " + class_name)
    plt.close('all')
    plt.rcParams["figure.figsize"] = [20, 2]
    plt.bar(list(output.keys()), list(
        output.values()), width=50)  # type: ignore
    plt.yticks([1])
    plt.xlabel("время, c")
    plt.ylabel("событие")
    plt.savefig(f'./net/images/events/{tt}+{class_name}.png')


# timestamps = timestamps[66:120] #78 #169
# timestamps = timestamps[78:190] #78 #169
