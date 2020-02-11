import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = data.load_data()

classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', "Shirt", 'Sneaker', 'Bag', 'Ankle boot']

trainImages = trainImages/255
testImages = testImages/255

print(trainImages[7])

plt.imshow(trainImages[7], cmap=plt.cm.binary)
plt.show()