#Xây dựng ứng dụng dự đoán chất lượng sữa với các thuộc tính:
#1.pH: Độ PH
#2.Temprature: Nhiệt độ
# 3.Taste: Vị
# 4.Odor: Mùi
# 5.Fat: Béo
# 6.Turbidity: Độ đục
# 7.Colour: Màu sắc
# 8.Grade: Chất lượng sữa

from numpy import double                                                            #Sử dụng kiểu dl double
import pandas as pd
from sklearn.naive_bayes import GaussianNB                                          #import thư viện naive_bayes
from sklearn.cluster import KMeans                                                  #import thư viện Kmeans
from sklearn.metrics import precision_score                                         #import hàm dự tính độ chính xác
import matplotlib.pyplot as plt                                                     #import thư viện vẽ biểu đồ
from tkinter import *                                                               #import thư viện hiển thị giao diện
import numpy as np
import statistics
import math

#---------------------------------------------------------------------------------

def variance(x,mean, n):
    return (math.pow((x-mean), 2)) / (n)

def Gauss(vari, x, mean):
    return (1/math.sqrt(2*math.pi* vari)) * math.exp(- ( math.pow(x - mean, 2) / (2*vari)))

def predictValue(data,dataMean, dataVarian, dicY, LenY):
    dic = {}
    for j in dicY:
        # print("stt----------------")
        arr = []
        for i in range(len(data)):

            # P = (dicY.get(j)) / LenY
            # print(P)

            # print((dataVarian.get(i)).get(j))
            # print((dataMean.get(i)).get(j))
            # print(data[i])
            # print((dicY.get(j)) / LenY)
            # print(Gauss((dataVarian.get(i)).get(j) ,(dataMean.get(i)).get(j) , data[i] ))
            arr.append(Gauss((dataVarian.get(i)).get(j) ,(dataMean.get(i)).get(j) , data[i] ))
        dic[j] = arr
    result = {}
    for y in dicY:
        P = (dicY.get(j)) / LenY
        for i in dic.get(y):
            P = P * i
        result[y] = P
    max = 0
    Rsclass = 0
    for i in result:
        if result.get(i) > max:
            max = result.get(i)
            Rsclass = i
    return Rsclass

def countValue(X, properties, Y,):

    dicY = {}
    for valueY in Y:                                      #Đếm số lần xuất hiện của nhãn thuộc tính phân lớp
        if valueY in dicY:
            dicY[valueY] += 1
        else:
            dicY[valueY] = 1

    dataMean = {}
    dataVarian = {}
    for propertie in range(properties):             #lặp qua các thuộc tính
        # print("TT: ", propertie)
        dic = {}
        dicVarri = {}
        dt = []
        for valueX in X:
            dt.append(valueX[propertie])             #Lấy các giá trị của thuộc tính
        for yitm in set(Y):
            temp = 0
            count = 0
            for y in range(len(Y)):
                for x in range(len(dt)):
                    if yitm == Y[y] and y == x:
                        temp += dt[x]
                        count += 1
            
            tb = temp/dicY.get(yitm)
            dic[yitm] = tb
        dataMean[propertie] = dic
        for yitm in set(Y):
            vari = 0
            for y in range(len(Y)):
                for x in range(len(dt)):
                    if yitm == Y[y] and y == x:
                        vari += variance(dt[x], (dataMean.get(propertie)).get(yitm),dicY.get(yitm))
            dicVarri[yitm] = vari
        dataVarian[propertie] = dicVarri
   
    return dataMean,dataVarian, dicY


#------------Nhập liệu và Tiền xử lý dữ liệu-----------------

DataTrain=pd.read_csv("milk_train.csv")
Xtrain=DataTrain.drop('Grade',axis=1)
YtrainDf=DataTrain["Grade"].values
Ytrain = []
# Ytrain = DataTrain["Class"].values  
for i in YtrainDf:
    if i < 0.5:
        Ytrain.append(0)
    elif i < 1:
        Ytrain.append(1)
    else:
        Ytrain.append(2)


DataTest=pd.read_csv("milk_test.csv")
Xtest=DataTest.drop('Grade',axis=1)
YtestDf=DataTest["Grade"].values
Ytest = []
for i in YtestDf:
    if i < 0.5:
        Ytest.append(0)
    elif i < 1:
        Ytest.append(1)
    else:
        Ytest.append(2)

print("Dữ liệu train: ",len(Xtrain))
print("Dữ liệu test: ",len(Xtest))


#------------------------------NAIVE_BAYESIAN-----------------------------
#------------------------------Sử dụng thư viện--------------------------------

naiveModel = GaussianNB()
naiveModel.fit(Xtrain,Ytrain)
predictNave=naiveModel.predict(Xtest)

print("\n-------------------Thuật toán Naive_Bayes-----------------------")
print("\nGiá trị dự đoán (Thư viện): ")
precisionLb = round(precision_score(Ytest, predictNave, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionLb,"%\n")


##--------------------------Không sử dụng thư viện--------------------------------

dataMean,dataVarian, dicY = countValue(Xtrain.values, len(Xtrain.values[0]), Ytrain)                #lấy số thuộc tính
# print("\nTần suất xuất hiện:")
# for i in range(len(data)):
#     print("Thuộc tính: ",i, ", Tần suất giá trị: ",data[i])

# print(dataMean)
# print(dataVarian)

# data = [6,130,8]
dataArr = []
# Rsclass = predictValue(data,dataMean,dataVarian,dicY,len(Ytrain))

# print(Rsclass)
for i in Xtest.values:
    Rsclass = predictValue(i,dataMean,dataVarian,dicY,len(Ytrain))
    dataArr.append(Rsclass)
print(dataArr)
precisionLb = round(precision_score(Ytest, dataArr, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionLb,"%\n")