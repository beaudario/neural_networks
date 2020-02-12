import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = data.load_data()

classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', "Shirt", 'Sneaker', 'Bag', 'Ankle boot']

trainImages = trainImages/255
testImages = testImages/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(trainImages, trainLabels, epochs=5)

testLoss, testAcc = model.evaluate(testImages, testLabels)

print("Tested Acc: ", testAcc)