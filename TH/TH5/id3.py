import pandas as pd
from sklearn import tree
from sklearn import metrics

Data=pd.read_csv("iris.csv")
print(Data)
X=Data.drop('Class',axis=1)
Y=Data["Class"].values
model=tree.DecisionTreeClassifier()
model.fit(X,Y)
kq=model.predict(X)
print(kq)
print(metrics.classification_report(Y,kq))
