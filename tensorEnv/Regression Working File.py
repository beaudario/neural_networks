import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "sex", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

xTrain, xTest, yTrain,  yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(xTrain, yTrain)
acc = linear.score(xTest, yTest)

print(acc)
print("Co: \n", linear.coef_)
print("Intercept: \n",  linear.intercept_)

predictions = linear.predict(xTest)

for x in range(len(predictions)):
    print(predictions[x], xTest[x], yTest[x])