import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

for x in range(1, 5):
    try:
        # Read the image in grayscale mode
        img = cv.imread(f"{x}.png", cv.IMREAD_GRAYSCALE)

        # Check if the image was correctly loaded
        if img is None:
            raise ValueError(f"Image {x}.png not found or unable to load.")

        # Resize the image to 28x28 if it's not already
        if img.shape != (28, 28):
            img = cv.resize(img, (28, 28))

        # Invert the colors
        img = np.invert(np.array([img]))

        # Normalize the image
        img = tf.keras.utils.normalize(img, axis=1)

        prediction = model.predict(img)
        print("----------------")
        print("The predicted value is:", np.argmax(prediction))
        print("----------------")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
