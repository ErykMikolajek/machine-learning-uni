# %%
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# %%
X_train = X_train / 255.0
X_test = X_test / 255.0

# %%
import matplotlib.pyplot as plt
plt.imshow(X_train[142], cmap="binary")
plt.axis('off')
plt.show()

# %%
class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]

# %%
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(300))
model.add(layers.Dense(100))
model.add(layers.Dense(10, activation="softmax"))

# %%
model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)

# %%
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %%
from datetime import datetime
def tensorboard_callback(name):
    dir_name = name + '/' + datetime.now().strftime("%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=dir_name, profile_batch=5)

# %%
model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_callback('./image_logs')])

# %%
import numpy as np
image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()

# %%
model.save("fashion_clf.h5")

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()

# %%
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# import pandas as pd
# housing_X = pd.DataFrame(X_train, columns=housing.feature_names)
# housing_X.isnull().any()

# %%
def reg_tensorboard_callback(name):
    dir_name = './housing_logs/' + name
    return tf.keras.callbacks.TensorBoard(log_dir=dir_name, profile_batch=5)

# %%
model_reg = keras.models.Sequential([
    layers.Dense(30),
    layers.Dense(1)
])
model_reg.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(clipvalue=1.0))

# %%
def early_stopping():
    return tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)

# %%
model_reg.fit(X_train, y_train, epochs=100, 
    validation_data=(X_valid, y_valid), callbacks=[early_stopping(), reg_tensorboard_callback('30')])

# %%
model_reg.save("reg_housing_1.h5")

# %%
model_reg_2 = keras.models.Sequential([
    layers.Dense(15, activation="relu"),
    layers.Dense(15, activation="relu"),
    layers.Dense(1)
])
model_reg_2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(clipvalue=1.0))
model_reg_2.fit(X_train, y_train, epochs=100,
    validation_data=(X_valid, y_valid), callbacks=[early_stopping(), reg_tensorboard_callback('15_15')])
model_reg_2.save("reg_housing_2.h5")

# %%
model_reg_3 = keras.models.Sequential([
    layers.Dense(30),
    layers.Dense(30),
    layers.Dense(1)
])
model_reg_3.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(clipvalue=1.0))
model_reg_3.fit(X_train, y_train, epochs=100,
    validation_data=(X_valid, y_valid), callbacks=[early_stopping(), reg_tensorboard_callback('30_30')])
model_reg_3.save("reg_housing_3.h5")

# %%
model_reg_4 = keras.models.Sequential([
    layers.Dense(15),
    layers.Dense(1)
])
model_reg_4.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(clipvalue=1.0))
model_reg_4.fit(X_train, y_train, epochs=100, 
    validation_data=(X_valid, y_valid), callbacks=[early_stopping(), reg_tensorboard_callback('15')])

# %%
model_reg.weights

# %%



