import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

Data=pd.read_csv("newMilk2.csv")
X=Data.drop('Grade',axis=1)
Y=Data["Grade"].values

# split data to training and testing 
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, random_state=42)

#id3
id3Model=tree.DecisionTreeClassifier()
id3Model.fit(trainX,trainY)
predictId3=id3Model.predict(testX)

#naive_bayes
naveModel = GaussianNB()
naveModel.fit(trainX,trainY)
predictNave=naveModel.predict(testX)

print("Dự đoán bằng thuật toán ID3: ")
print(metrics.confusion_matrix(testY,predictId3))
print(metrics.classification_report(testY,predictId3))
precision = round(precision_score(testY, predictId3, average='micro') * 100,2)
print("Độ chính xác precision : ", precision,"%\n")


print("Dự đoán bằng thuật toán Naive_bayes: ")
print(metrics.confusion_matrix(testY,predictNave))
print(metrics.classification_report(testY,predictNave))
precision = round(precision_score(testY, predictNave, average='micro') * 100,2)
print("Độ chính xác precision : ", precision,"%\n")



