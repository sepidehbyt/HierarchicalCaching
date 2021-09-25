import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# build neural networks by transfer learning using model A
model_A = keras.models.load_model("my_model_A.h5")
# we gotta clone model A
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# freeze model_A layers
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# You must always compile your model after you freeze or unfreeze layers.
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# train the model for a few epochs, then unfreeze the reused layers (which requires compiling the model again)
# After unfreezing the reused layers, it is usually a good idea to reduce the learning rate, once again to avoid
# damaging the reused weights
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

# the default lr is 1e-2
optimizer = keras.optimizers.SGD(lr=1e-4)
# compile again
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = np.argmax(model.predict(X_new), axis=1)
