# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
from scipy.stats import reciprocal

param_distribs = {
"model__n_hidden": range(4),
"model__n_neurons": range(1, 101),
"model__learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
"model__optimizer": ['adam', 'sgd', 'nesterov']
}

# %%
import tensorflow as tf
from tensorflow import keras

def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(X_train.shape[1]))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
    model.add(keras.layers.Dense(1))
    if optimizer == 'sgd':
        my_optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipvalue=0.9)
    elif optimizer == 'nesterov':
        my_optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, clipvalue=0.9)
    elif optimizer == 'adam':
        my_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.9)
    else:
        my_optimizer = keras.optimizers.get(optimizer, learning_rate=learning_rate, clipvalue=0.9)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model

# %%
import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])

# %%
from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)

# %%
import pickle

with open('rnd_search_params.pkl', 'wb') as params_pickle:
    pickle.dump(rnd_search_cv.best_params_, params_pickle)

with open('rnd_search_scikeras.pkl', 'wb') as scikeras_pickle:
    pickle.dump(rnd_search_cv, scikeras_pickle)

# %%
import keras_tuner as kt

def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "nesterov"])

    if optimizer == 'sgd':
        my_optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipvalue=0.9)
    elif optimizer == 'nesterov':
        my_optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, clipvalue=0.9)
    elif optimizer == 'adam':
        my_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.9)
    else:
        my_optimizer = keras.optimizers.get(optimizer, learning_rate=learning_rate, clipvalue=0.9)

    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(X_train.shape[1]))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer=my_optimizer)
    return model

# %%
random_search_tuner = kt.RandomSearch(build_model_kt, objective="val_loss", max_trials=10, overwrite=True,
directory="my_california_housing", project_name="my_rnd_search", seed=42)

# %%
import os

root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)

# %%
# %load_ext tensorboard
# %tensorboard --logdir {root_logdir}
random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tb, es])

# %%

params = random_search_tuner.get_best_hyperparameters()[0]

best_hyperparameters = {
    'n_hidden': params.get('n_hidden'),
    'n_neurons': params.get('n_neurons'),
    'learning_rate': params.get('learning_rate'),
    'optimizer': params.get('optimizer')
}

best_hyperparameters

# %%
with open('kt_search_params.pkl', 'wb') as params_keras:
    pickle.dump(best_hyperparameters, params_keras)

# %%
best_model = random_search_tuner.get_best_models()[0]
best_model.save('kt_best_model.h5')


