import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import config as cfg
from math import fabs
from GenerateTrainData import gen_train_data


t_observation = cfg.sys_observation_window_days


def train_mlp(scale):
    data = gen_train_data()
    return train_mlp_type(data, scale, 'irm')


def train_mlp_type(data, scale, train_type):
    if scale:
        mm_scale = MinMaxScaler()
        data = mm_scale.fit_transform(data)
    else:
        mm_scale = None

    x_train_full, x_test, y_train_full, y_test = train_test_split(data[:, 0:t_observation],
                                                                  data[:, t_observation])
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

    normalizer = preprocessing.Normalization(input_shape=[t_observation], axis=None)
    normalizer.adapt(data[:, 0:t_observation])

    model = keras.models.Sequential([normalizer])
    model.add(keras.layers.Dense(t_observation))
    model.add(keras.layers.BatchNormalization())
    if train_type == 'irm':
        model.add(keras.layers.Dense(45, activation="elu", kernel_initializer="he_normal"))
    else:
        model.add(keras.layers.Dense(90, activation="elu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    if train_type == 'irm':
        model.add(keras.layers.Dense(1, activation="softplus"))
    else:
        model.add(keras.layers.Dense(1, activation="sigmoid"))

    if train_type == 'irm':
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="mse", optimizer=optimizer)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    if train_type == 'irm':
        model.fit(x_train, y_train, epochs=32, batch_size=512, validation_data=(x_valid, y_valid),
                  callbacks=[early_stopping_cb])
    else:
        model.fit(x_train, y_train, epochs=32, batch_size=2, validation_data=(x_valid, y_valid),
                  callbacks=[early_stopping_cb])

    mse_test = model.evaluate(x_test, y_test, verbose=0)
    print(mse_test)

    return model, x_test, y_test, mm_scale


class MultilayerPerceptron:

    def __init__(self, data, scale, train_type):
        self.model, self.x_test, self.y_test, self.mm_scale = train_mlp_type(data, scale, train_type)
        # self.model, self.x_test, self.y_test, self.mm_scale = train_mlp(False)

    def predict(self, x_obs):
        y_pred = self.model.predict(x_obs)
        if self.mm_scale is not None:
            return self.mm_scale.inverse_transform(np.c_[x_obs, y_pred])[:, t_observation]
        else:
            return y_pred

    def print_test(self, num, accuracy):
        x_new = self.x_test[:num]
        y_pred = self.model.predict(x_new)

        pred = np.c_[x_new, y_pred]
        actual = np.c_[x_new, self.y_test[:num]]
        if self.mm_scale is not None:
            actual = np.round(self.mm_scale.inverse_transform(actual))
            pred = np.round(self.mm_scale.inverse_transform(pred))
        print(pred)
        print(actual)
        print('test actual')
        for i in range(num):
            if fabs(pred[i, t_observation] - actual[i, t_observation]) > accuracy:
                print(pred[i, t_observation], actual[i, t_observation])
