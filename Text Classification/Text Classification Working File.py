import tensorflow as tf
import numpy as np
from tensorflow import keras

data = keras.datasets.imdb

(trainData, trainLabels), (testData, testLabels) = data.load_data(num_words=10000)

print(trainData[0])

wordIndex = data.get_word_index()

wordIndex = {k: (v + 3) for k, v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<STARTS>"] = 1
wordIndex["<UNK>"] = 2
wordIndex["UNUSED"] = 3

reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

trainData = keras.preprocessing.sequence.pad_sequences(trainData, value=wordIndex["<PAD>"], padding="post", maxlen=250)
testData = keras.preprocessing.sequence.pad_sequences(testData, value=wordIndex["<PAD>"], padding="post", maxlen=250)


def decodeReview(text):
    return " ".join([reverseWordIndex.get(i, "?") for i in text])


# model down here

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


