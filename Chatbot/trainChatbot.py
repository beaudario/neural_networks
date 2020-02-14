import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


words = []
classes = []
documents = []
ignoreWords = ['!', '?']
dataFile = open('intents.json').read()
intents = json.loads(dataFile)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word an tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreWords]
words = sorted(list(set(classes)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initializing training data
training = []
outputEmpty = [0] * len(classes)

for doc in documents:

    # initializing bag of words
    bag = []

    # list of tokenized words for the pattern
    patternWords = doc[0]

    # lemmatize each word - create base word, in attempt to represent related words
    patternWords = [lemmatizer.lemmatize(word.lower()) for word in patternWords]

    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in patternWords else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    outputRow = list(outputEmpty)
    outputRow[classes.index(doc[1])] = 1

    training.append([bag, outputRow])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
trainX = list(training[:, 0])
trainY = list(training[:, 1])

print("TRAINING DATA CREATED")

# create model - 3 layers. first layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of
# neurons equal to number of intents to predict output intent of softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))

# compile model. Stochatic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("MODEL CREATED")
