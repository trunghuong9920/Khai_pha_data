import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics

Data=pd.read_csv("newMilk.csv")
print(Data)
X=Data.drop('Grade',axis=1)
Y=Data["Grade"].values
model= SVC()
model.fit(X,Y)
print(model)
kq=model.predict(X)
print(kq)
print(metrics.confusion_matrix(Y,kq))
print(metrics.classification_report(Y,kq))
