
# Import necessary libraries
import tensorflow as tf ** # TensorFlow library for machine learning**
from tensorflow import keras  # Keras API for building neural networks
from tensorflow.keras import layers  # Layers for building neural network architecture
import matplotlib.pyplot as plt  # Matplotlib library for visualizing results

# Load MNIST dataset and split it into training and testing sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1 range for better model performance
x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

import numpy as np

# Pick a random image
index = 777
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title("Actual Label: {}".format(y_test[index]))
plt.show()

# Predict
pred = model.predict(np.expand_dims(x_test[index], axis=0))
print("Predicted Label:", np.argmax(pred))
