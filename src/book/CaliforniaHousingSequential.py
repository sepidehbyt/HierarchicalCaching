import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# scaler = StandardScaler()
#
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.transform(X_valid)
# X_test = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")

run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_06_07-15_15_22'
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])

mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]  # pretend t
y_proba = model.predict(X_new)
y_proba.round(2)


# test_logdir = get_run_logdir()
# writer = tf.summary.create_file_writer(test_logdir)
# with writer.as_default():
#     for step in range(1, 1000 + 1):
#         tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
#         data = (np.random.randn(100) + 2) * step / 100  # some random data
#         tf.summary.histogram("my_hist", data, buckets=50, step=step)
#         images = np.random.rand(2, 32, 32, 3)  # random 32×32 RGB images
#         tf.summary.image("my_images", images * step / 1000, step=step)
#         texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
#         tf.summary.text("my_text", texts, step=step)
#         sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
#         audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
#         tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
#
#
# # create model in case of test and tune hyperparameters
# def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=None):
#     if input_shape is None:
#         input_shape = [8]
#     model = keras.models.Sequential()
#     model.add(keras.layers.InputLayer(input_shape=input_shape))
#     for layer in range(n_hidden):
#         model.add(keras.layers.Dense(n_neurons, activation="relu"))
#         model.add(keras.layers.Dense(1))
#         optimizer = keras.optimizers.SGD(lr=learning_rate)
#         model.compile(loss="mse", optimizer=optimizer)
#         return model
#
#
# # keras_regressor
# keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# keras_reg.fit(X_train, y_train, epochs=100,
#               validation_data=(X_valid, y_valid),
#               callbacks=[keras.callbacks.EarlyStopping(patience=10)])
# mse_test = keras_reg.score(X_test, y_test)
# y_pred = keras_reg.predict(X_new)
#
# # we want to test hundreds of states , instead of above code:
# # RandomizedSearchCV uses K-fold cross-validation
# param_distribs = {"n_hidden": [0, 1, 2, 3],
#                   "n_neurons": np.arange(1, 100),
#                   "learning_rate": reciprocal(3e-4, 3e-2),
#                   }
# rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
# rnd_search_cv.fit(X_train, y_train, epochs=100,
#                   validation_data=(X_valid, y_valid),
#                   callbacks=[keras.callbacks.EarlyStopping(patience=10), tensorboard_cb])
# model = rnd_search_cv.best_estimator_.model
# mse_test = model.evaluate(X_test, y_test)
#
# # some optimization: momentum
# optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
# # add NAG
# optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
# # RMS prop (AdaGrad)
# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
# # Adam and Nadam
# optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#
# # If these techniques remain insufficient, check out the TensorFlow  Model Optimization Toolkit (TF-MOT),
# # which provides a pruning API capable of iteratively removing connections during training based on their magnitude.
#
# # Learning rate scheduling
# # power scheduling
# optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
#
#
# # Exponential scheduling and piecewise scheduling
# def exponential_decay(lr0, s):
#     def exponential_decay_fn(epoch):
#         return lr0 * 0.1 ** (epoch / s)
#
#     return exponential_decay_fn
#
#
# exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
#
# # LearningRateScheduler callback, giving it the schedule function, and pass this callback to the fit() method
# lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])
#
#
# # When you save a model, the optimizer and its learning rate get saved along with it
#
# # piecewise constant scheduling
# def piecewise_constant_fn(epoch):
#     if epoch < 5:
#         return 0.01
#     elif epoch < 15:
#         return 0.005
#     else:
#         return 0.001
#
#
# # performance scheduling
# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# # how to implement the same exponential schedule as the exponential_decay_fn() function defined earlier:
# s = 20 * len(X_train) // 32  # number of steps in 20 epochs (batch size =32)
# learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
# optimizer = keras.optimizers.SGD(learning_rate)
#
# # regularization
# # apply l2 regularization to a Keras layer’s connection weights, using a regularization factor of 0.01
# layer = keras.layers.Dense(100, activation="elu",
#                            kernel_initializer="he_normal",
#                            kernel_regularizer=keras.regularizers.l2(0.01))
#
# # to avoid repeating stuff:
# from functools import partial
# RegularizedDense = partial(keras.layers.Dense, activation="elu",
#                            kernel_initializer="he_normal",
#                            kernel_regularizer=keras.regularizers.l2(0.01))
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     RegularizedDense(300),
#     RegularizedDense(100),
#     RegularizedDense(10, activation="softmax", kernel_initializer="glorot_uniform")
# ])
#
# # use dropout
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(10, activation="softmax")
# ])
#
# # use MC dropout
# y_probas = np.stack([model(X_test_scaled, training=True) for sample in range(100)])
# y_proba = y_probas.mean(axis=0)
#
