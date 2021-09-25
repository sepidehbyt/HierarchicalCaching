import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# input_ = keras.layers.Input(shape=X_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.Concatenate()([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.Model(inputs=[input_], outputs=[output])

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid))

mse_test = model.evaluate((X_test_A, X_test_B), y_test)

y_pred = model.predict((X_new_A, X_new_B))


# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
#
# mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3]  # pretend t
y_proba = model.predict(X_new)
y_proba.round(2)

# to save the model in hdf5
model.save("my_keras_model.h5")

# retrieve model
model = keras.models.load_model("my_keras_model.h5")

# after build and compile model
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# or use early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# role back to best model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb])
# if we didn't use earlyStopping we would need to restore
model = keras.models.load_model("my_keras_model.h5")


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
    # also n_train_begin() ,
    # on_train_end() , on_epoch_begin() , on_epoch_end() ,
    # on_batch_begin() , and on_batch_end() .
    # during evaluation on_test_begin() ,
    # on_test_end() , on_test_batch_begin() , or on_test_batch_end()
    # (called by evaluate() )
    # during prediction on_predict_begin() , on_predict_end() ,
    # on_predict_batch_begin() , or on_predict_batch_end()
    # (called by predict() ).
