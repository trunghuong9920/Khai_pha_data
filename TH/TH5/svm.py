import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score


Data=pd.read_csv("newMilk.csv")
X=Data.drop('Grade',axis=1)
Y=Data["Grade"].values
model= SVC()
model.fit(X,Y)
kq=model.predict(X)
print(kq)
print(metrics.confusion_matrix(Y,kq))
print(metrics.classification_report(Y,kq))

precisionSVM = round(precision_score(Y, kq, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionSVM,"%\n")

