from pyexpat import model
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score


data = pd.read_csv("diabetes.csv")
data.head

data = data.replace("tested_positive", 0)
data = data.replace("tested_negative", 1)
X = data.drop("class", axis=1)
Y = data["class"].values

model = KMeans(n_clusters=2).fit(X)

precision = round(precision_score(Y, model.predict(X), average='micro') * 100,2)

print(Y)
print(model.predict(X))
print(metrics.confusion_matrix(Y,model.predict(X)+1))
print("Độ chính xác= ",precision)