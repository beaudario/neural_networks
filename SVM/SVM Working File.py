import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(xTrain, yTrain)
classes = ["malignant", "benign"]