import tensorflow as tf
import numpy as np
from tensorflow import keras

data = keras.datasets.imdb

(trainData, trainLabels), (testData, testLabels) = data.load_data(num_words=88000)

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


"""
# model down here

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

xVal = trainData[:10000]
xTrain = trainData[10000:]

yVal = trainLabels[:10000]
yTrain = trainLabels[10000:]

fitModel = model.fit(xTrain, yTrain, epochs=40, batch_size=512, validation_data=(xVal, yVal), verbose=1)

results = model.evaluate(testData, testLabels)
print(results)

model.save("model.h5")
"""


def reviewEncode(s):
    encoded = [1]

    for word in s:
        if word.lower() in wordIndex:
            encoded.append(wordIndex[word.lower()])
        else:
            encoded.append(2)

    return encode


model = keras.models.load_model("model.h5")

with open("testReview.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nLine = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "")\
            .replace('"', '').strip()
        encode = reviewEncode(nLine)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=wordIndex["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

"""
testReview = testData[0]
predict = model.predict(testReview)

print("Review: ")
print(decodeReview(testReview))
print("Prediction: ", str(predict[0]))
print("Actual: ", str(testLabels[0]))
print(results)
"""
