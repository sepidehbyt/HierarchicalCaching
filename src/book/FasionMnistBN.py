import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# build neural networks
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

# adding the BN layers after the activation functions
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation("elu"),
#     keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation("elu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

# hyperparameters for BN : momentum, axis

# gradient clipping
# This optimizer will clip every component of the gradient vector to a value between –1.0 and 1.0.
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)
# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = np.argmax(model.predict(X_new), axis=1)
