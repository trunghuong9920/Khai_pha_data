import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

Data=pd.read_csv("iris.csv")
print(Data)
X=Data.drop('Class',axis=1)
Y=Data["Class"].values
model= GaussianNB()
model.fit(X,Y)
print(model)
kq=model.predict(X)
print(kq)
print(metrics.confusion_matrix(Y,kq))
print(metrics.classification_report(Y,kq))
