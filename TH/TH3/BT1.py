from tkinter import Y
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('glass.csv')

diabetes = diabetes.replace("'build wind float'", 0)
diabetes = diabetes.replace("'vehic wind float'", 1)
diabetes = diabetes.replace('headlamps', 2)
diabetes = diabetes.replace('containers', 3)
diabetes = diabetes.replace("'build wind non-float'", 4)
diabetes = diabetes.replace('tableware', 5)

X = diabetes.drop("Type",axis=1)
Y = diabetes["Type"].values

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, random_state=42)

model = LinearRegression()
model.fit(trainX,trainY)

y_pred = model.predict(testX)

print("Hệ số hồi quy: ", model.coef_)
print("Sai số hồi quy: ",model.intercept_)
print("Dự đoán: ", y_pred)
print(pd.DataFrame({"Name":X.columns,"Hệ số hồi quy": model.coef_}).sort_values(by="Hệ số hồi quy"))
