import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

filename = "linear_regression_model.pickle"



def multiple_times_training(times):
    best = 0
    best_model = None
    for i in range(times):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        model = LinearRegression()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test) * 100
        if accuracy > best:
            best = accuracy
            best_model = model
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
    return best_model



data = pd.read_csv("student-mat.csv", sep=";")
data = data[["failures", "absences", "studytime", "G1", "G2", "G3"]]
predict = "G3"
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

try:
    with open(filename, 'rb') as f:
        model = pickle.load(f)

except IOError:
    model = multiple_times_training(100)

acc = model.score(x_test, y_test) * 100


print(f"Accuracy of predicting: {acc:.4f}%")
