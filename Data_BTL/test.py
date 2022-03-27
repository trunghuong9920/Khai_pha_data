from unittest import result
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

Data=pd.read_csv("testdata.csv")
X=Data.drop('Giaban',axis=1)
Y=Data["Giaban"].values

def countValue(propertie,value,X, classNumber,Y):
    ixCheck = 0
    dic = {}
    for vl in X:
        if(vl[propertie] == value):             #Kiểm tra giá trị đang xét với giá trị trong bảng
            ixCount = 0
            for valueY in Y:
                if(ixCount == ixCheck):                #Lấy ra vị trí của Y tương ứng X đang xét
                    if Y[valueY] in dic:               #Đếm tần xuất xuất hiện của Y tương ứng với X đang xét
                        dic[Y[valueY]] += 1
                    else:
                        dic[Y[valueY]] = 1
                ixCount += 1
        ixCheck +=1
    data = []
    for i in range(0,classNumber):
        data.append(dic.get(i) if dic.get(i) != None else 0)
    return data
        
def predictValue(dataCount,data, Y,classNumber):
    dicY = {}
    for valueY in Y:
        if valueY in dicY:
            dicY[valueY] += 1
        else:
            dicY[valueY] = 1
    result = {}
    for i in range(0,classNumber):
        for j in range(len(data)):
            for x in range(len(dataCount)):
                if(j == x):
                    print(data[j], dataCount[x])
            

def bayesian(X, properties, Y):
    data = []
    for propertie in range(properties):
        dicProperties = {}
        for valueX in X:
            dicProperties[valueX[propertie]] = countValue(propertie,valueX[propertie],X,len(set(Y)),Y)        #Tính tần suất xuất hiện
        data.append(dicProperties)
    return data


classNumber = len(set(Y))

data = bayesian(X.values, len(X.values[0]), Y)                #lấy số thuộc tính

print("\nTần suất xuất hiện:")
for i in data:
    print(i)

dataPredict = [1,0,1]
predictValue(data,dataPredict,Y,classNumber)
