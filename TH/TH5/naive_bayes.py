import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score  # import hàm dự tính độ chính xác


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
precisionLb = round(precision_score(
    Y, kq, average='micro') * 100, 2)
print("Độ chính xác precision : ", precisionLb, "%\n")
