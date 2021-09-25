import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import config as cfg
from GenerateTrainData import gen_train_data

from GenerateRequests import generate_sim_requests
from GenerateTrainData import turn_requests_into_observation_window
from GenerateTrainData import turn_observation_into_w_s
from GenerateTrainData import turn_observation_into_irm
from Graphs import plot_delivery_type_per_window, plot_content_type_uncached
from BS import BaseStationThesis
from BS import BaseStationPaper
from CU import ControlUnit
from CU import CloudUnit


root_logdir = os.path.join(os.curdir, "my_logs")
t_observation = cfg.sys_observation_window_days


def print_test(model, scaler, X_test, y_test, num):
    X_new = X_test[:num]
    y_proba = model.predict(X_new)

    whateve = np.c_[X_new, y_proba]
    whateve1 = np.c_[X_new, y_test[:num]]
    whateve1 = np.round(scaler.inverse_transform(whateve1))
    whateve = np.round(scaler.inverse_transform(whateve))
    print(whateve)
    print(whateve1)
    print('test actual')
    for i in range(num):
        print(whateve[i, t_observation], whateve1[i, t_observation])


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == '__main__':

    tot_requests = generate_sim_requests(20, 2)
    tot_requests = np.array(tot_requests)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    total_observation_window = turn_requests_into_observation_window(tot_req_count, tot_requests, 20)
    data = turn_observation_into_w_s(total_observation_window, 5, 20)

    # data = gen_train_data()

    scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)

    normalizer = preprocessing.Normalization(input_shape=[t_observation], axis=None)
    normalizer.adapt(data[:, 0:t_observation])

    X_train_full, X_test, y_train_full, y_test = train_test_split(data[:, 0:t_observation], data[:, t_observation])
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    # scaler = StandardScaler()
    #
    # X_train = scaler.fit_transform(X_train)
    # X_valid = scaler.transform(X_valid)
    # X_test = scaler.transform(X_test)


    def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, optimizer=keras.optimizers.SGD,
                    activation_function='softplus', input_shape=None):
        if input_shape is None:
            input_shape = [t_observation]
        model = keras.models.Sequential([normalizer])
        # model.add(keras.layers.Dense(t_observation, activation="relu"))
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(1, activation=activation_function))
        # model.add(keras.layers.Dense(1))
        optimizer = optimizer(learning_rate=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model


    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
    # keras_reg.fit(X_train, y_train, epochs=32,
    #               validation_data=(X_valid, y_valid),
    #               callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    # mse_test = keras_reg.score(X_test, y_test)
    # y_pred = keras_reg.predict(X_new)

    # we want to test hundreds of states , instead of above code:
    # RandomizedSearchCV uses K-fold cross-validation
    param_distribs = {"n_hidden": [0, 1, 2, 3],
                      "n_neurons": np.arange(1, 100),
                      "optimizer": [keras.optimizers.SGD, keras.optimizers.Adam],
                      "activation_function": ["softplus", "sigmoid", "linear", ],
                      "learning_rate": reciprocal(3e-4, 3e-2),
                      }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=100, cv=3)

    # run_logdir = get_run_logdir()
    # tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    rnd_search_cv.fit(X_train, y_train, epochs=32,
                      validation_data=(X_valid, y_valid),
                      # callbacks=[keras.callbacks.EarlyStopping(patience=10), tensorboard_cb],
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)],
                      verbose=0)
    model = rnd_search_cv.best_estimator_.model
    mse_test = model.evaluate(X_test, y_test, verbose=0)

    print(mse_test)
    print(rnd_search_cv.best_params_)

    # # model = keras.models.Sequential()
    # # # model.add(keras.layers.InputLayer(input_shape=[t_observation]))
    # # model.add(keras.layers.Dense(t_observation, activation="relu"))
    # # model.add(keras.layers.Dense(30, activation="relu"))
    # # model.add(keras.layers.Dense(10, activation="relu"))
    # # model.add(keras.layers.Dense(1, activation="softmax"))
    #
    # # model.compile(loss="mse", optimizer=optimizer)
    # # model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
    # optimizer = keras.optimizers.SGD(learning_rate=1e-2)
    # model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
    #
    # # run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_06_07-15_15_22'
    # # tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    # early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    #
    # history = model.fit(X_train, y_train,
    #                     epochs=32, validation_data=(X_valid, y_valid),
    #                     callbacks=[early_stopping_cb])
    #                     # callbacks=[early_stopping_cb, tensorboard_cb])
    #
    # mse_test = model.evaluate(X_test, y_test)
