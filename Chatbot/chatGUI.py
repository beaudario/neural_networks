import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np

from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def CleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
    return sentenceWords


# return bag of words array: 0 to 1 for each word in the bag that exists in the sentence
def Bow(sentence, words, showDetails=True):
    # tokenize the pattern
    sentenceWords = CleanUpSentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)

    for s in sentenceWords:
        for i, w in enumerate(words):
            if w == s:

                # assign 1 if current word is in the vocabulary position
                bag[i] = 1

                if showDetails:
                    print('Found in bag: %s' % w)

    return np.array(bag)


def PredictClass(sentence, model):
    # filter out predictions below a treshold
    p = Bow(sentence, words, showDetail=True)
    res = model.predict(np.array([p]))[0]
    ERROR_TRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []

    for r in results:
        returnList.append({"intent": classes[r[0]], "probabilty": str(r[1])})

    return returnList


def GetResponse(ints, intentsJson):
    tag = ints[0]['intent']
    listOfIntents = intentsJson['intents']

    for i in listOfIntents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break

    return result


def ChatbotResponse(msg):
    ints = PredictClass(msg, model)
    res = GetResponse(ints, intents)

    return res


# creating GUI with tkinter
import tkinter
from tkinter import *


def Send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = ChatbotResponse(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# create chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

# bind scroll bar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# create button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97",
                    activebackground="#3c9d9b", fg="#ffffff", command=Send)

# create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

# place all comments on screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
