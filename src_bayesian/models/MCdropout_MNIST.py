from tensorflow.keras import datasets
from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(300, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(300, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation="softmax"))

optimizer = optimizers.Nadam(lr=0.001)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=20)
model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['loss'])

def predict_proba(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.stack(preds).mean(axis=0)

def predict_class(X, model, num_samples):
    proba_preds = predict_proba(X, model, num_samples)
    return np.argmax(proba_preds, axis=1)

y_pred = predict_class(X_test, model, 100)
acc = np.mean(y_pred == y_test)

## predicted probabilities
y_pred_proba = predict_proba(X_test, model, 100)

softmax_output = np.round(model.predict(X_test[1:2]), 3)
mc_pred_proba = np.round(y_pred_proba[1], 3)
print(softmax_output, mc_pred_proba)

## ==================== regression problem ====================
(X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()

def get_dropout(input_tensor, p=0.1, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

inp = Input(13,)
x = Dense(64, activation="relu")(inp)
x = get_dropout(x, p=0.1, mc='mc')
x = Dense(32, activation="relu")(x)
x = get_dropout(x, p=0.1, mc='mc')
x = Dense(32, activation="relu")(x)
x = get_dropout(x, p=0.1, mc='mc')
out = Dense(1, activation="relu")(x)
model = Model(inputs=inp, outputs=out)

optimizer = optimizers.Nadam(lr=0.001)
model.compile(loss="mse", optimizer=optimizer)

model.fit(X_train, y_train, epochs=100, validation_split=0.1)

def predict_dist(X, model, num_samples) :
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.hstack(preds)

def predict_point(X, model, num_samples):
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist.mean(axis=1)

y_pred_dist = predict_dist(X_test, model, 100)
y_pred = predict_point(X_test, model, 100)

sns.kdeplot(y_pred_dist[0], shade=True)
plt.axvline(y_pred[0], color='red')
plt.show()



