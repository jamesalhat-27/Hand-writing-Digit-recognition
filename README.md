# Handwritten Digit Recognition using CNN (MNIST Dataset)

A deep learning project to recognize handwritten digits (0–9) using the MNIST dataset and Convolutional Neural Networks (CNN) built with TensorFlow/Keras.

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

##  Project Objective

The goal of this project is to:
- Build a CNN-based deep learning model that can recognize handwritten digits from images.
- Train the model on the MNIST dataset and evaluate its performance.
- Visualize predictions and model metrics.
- Understand how CNNs can be applied in real-world digit recognition systems.

---

##  Tools & Technologies

- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn  
- **Dataset:** MNIST (from `tensorflow.keras.datasets`)  
- **IDE:** Jupyter Notebook / Google Colab  

---

##  Dataset Details

- **Name:** MNIST (Modified National Institute of Standards and Technology)  
- **Images:** 28x28 grayscale  
- **Training Set:** 60,000 images  
- **Test Set:** 10,000 images  
- **Labels:** Digits from 0 to 9  

---

##  How to Run the Project

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition

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

#Model Architecture (CNN)
Conv2D → MaxPooling2D → Conv2D → MaxPooling2D

Flatten → Dense → Dropout → Dense (Softmax)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])
Results
Test Accuracy: ~98%

Loss: Very low, indicating good generalization

Applications
Bank cheque processing

Postal code recognition

Form digitization

Educational tools (automated grading)

License
This project is open-source and available under the MIT License.

 Contact
If you have any questions or suggestions, feel free to reach out!
Name: James Alhat
Email: jamesalhat3@gmail.com
GitHub: jamesalhat-27
