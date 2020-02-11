import sklearn
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benign"]

clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(xTrain, yTrain)

yPred = clf.predict(xTest)

acc = metrics.accuracy_score(yTest, yPred)

print(acc)