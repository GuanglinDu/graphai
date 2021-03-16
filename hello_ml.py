# A Linear Regression ML demo, created on Mar 13, 2021
# Say hello to the "Hello, World" of machine learning
# See https://developers.google.com/codelabs/tensorflow-1-helloworld
# python -m venv --system-site-packages .\venv
# .\venv\Script\activate

# python -m pip install --upgrade pip

# pip install --upgrade tensorflow

# Fix error: https://github.com/tensorflow/tensorflow/issues/37186
# Attempting to fetch value instead of handling error Internal:
# Could not retrieve CU DA device attribute (81: UNKNOWN ERROR (1) ...
# pip install tensorflow-cpu

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and compile the neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data (y = 3x + 1)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#  Train the neural network
model.fit(xs, ys, epochs=500)

# Use the model
print(model.predict([5.0]))
print(model.predict([10.0]))

