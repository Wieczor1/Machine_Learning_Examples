import pickle

import sklearn
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

filename = "logistic_regression_model.pickle"

def multiple_times_training(times):
    best = 0
    best_model = None
    for i in range(times):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        model = sklearn.linear_model.LogisticRegression()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test) * 100
        if accuracy > best:
            best = accuracy
            best_model = model
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
    return best_model

data = pd.read_csv("car.data")

# print(data.head())
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)  # encoding values so everything is numerical
X = list(data.drop(columns='class').itertuples(index=False, name=None))  # tuple of values to train on
y = list(data["class"])  # values to predict
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


try:
    with open(filename, 'rb') as f:
        logistic = pickle.load(f)

except IOError:
    logistic = multiple_times_training(100)


log_accuracy = logistic.score(x_test, y_test) * 100
print(f"Logistic regression accuracy: {log_accuracy:.4f}%")


neighbors = KNeighborsClassifier(n_neighbors=11)

neighbors.fit(x_train, y_train)
accuracy = neighbors.score(x_test, y_test) * 100
print(f"K nearest neighbors algorithm accuracy: {accuracy:.4f}%")

clf = svm.SVC(kernel="rbf", C=4)
clf.fit(x_train, y_train)
svc_accuracy = clf.score(x_test, y_test) * 100
print(f"SVC accuracy: {svc_accuracy:.4f}%")
