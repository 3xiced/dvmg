from keras.models import Sequential
from keras.layers import Dense, Input
from .base import ModelBase

from typing import Any, Union
from tensorflow import Tensor
import tensorflow as tf


class l2i14(ModelBase):
    """2 скрытых слоя, 14 входов - реконструкции weight center, quantile, etc., sigmoid, sigmoid

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(14,)))
        self._model.add(Dense(70, activation='sigmoid', name='hidden_1'))
        self._model.add(Dense(32, activation='sigmoid', name='hidden_2'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))


class l2i16(ModelBase):
    """2 скрытых слоя, 16 входов - реконструкции octante, etc., sigmoid, sigmoid

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(16,)))
        self._model.add(Dense(70, activation='sigmoid', name='hidden_1'))
        self._model.add(Dense(32, activation='sigmoid', name='hidden_2'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))


class l3i14(ModelBase):
    """3 скрытых слоя, 14 входов - реконструкции octante, etc., relu, relu, relu

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(14,)))
        self._model.add(Dense(70, activation='relu', name='hidden_1'))
        self._model.add(Dense(32, activation='relu', name='hidden_2'))
        self._model.add(Dense(16, activation='relu', name='hidden_3'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))


class l3i42(ModelBase):
    """3 скрытых слоя, 42 входа - 3 реконструкции quantile/weightcenter, etc., sigmoid, sigmoid, sigmoid\n
    Модель для множественной параллельной реконструкции в 3 их разновидности

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(42,)))
        self._model.add(Dense(102, activation='sigmoid', name='hidden_1'))
        self._model.add(Dense(56, activation='sigmoid', name='hidden_2'))
        self._model.add(Dense(24, activation='sigmoid', name='hidden_3'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))


class l3i28(ModelBase):
    """3 скрытых слоя, 28 входа - 3 реконструкции quantile/weightcenter, etc., sigmoid, sigmoid, sigmoid\n
    Модель для множественной параллельной реконструкции в 3 их разновидности

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(28,)))
        self._model.add(Dense(102, activation='sigmoid', name='hidden_1'))
        self._model.add(Dense(56, activation='sigmoid', name='hidden_2'))
        self._model.add(Dense(24, activation='sigmoid', name='hidden_3'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))


class l4i56(ModelBase):
    """4 скрытых слоя, 56 входа - 3 реконструкции quantile/weightcenter, etc., sigmoid, sigmoid, sigmoid\n
    Модель для множественной параллельной реконструкции в 3 их разновидности

    Args:
        ModelBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self._model: Sequential = Sequential()
        self._model.add(Input(shape=(56,)))
        self._model.add(Dense(256, activation='sigmoid', name='hidden_1'))
        self._model.add(Dense(128, activation='sigmoid', name='hidden_2'))
        self._model.add(Dense(64, activation='sigmoid', name='hidden_3'))
        self._model.add(Dense(32, activation='sigmoid', name='hidden_4'))
        self._model.add(Dense(7, activation='softmax', name='output'))

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, x_train: Union[Tensor, list], y_train: Union[Tensor, list], batch_size: int = 20, epochs: int = 5) -> None:
        self._model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, data: Tensor) -> Any:
        return self._model.predict(tf.expand_dims(data, axis=0))
