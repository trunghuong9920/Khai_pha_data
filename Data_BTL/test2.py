#Xây dựng ứng dụng dự đoán chất lượng sữa với các thuộc tính:
#1.pH: Độ PH
#2.Temprature: Nhiệt độ
# 3.Taste: Vị
# 4.Odor: Mùi
# 5.Fat: Béo
# 6.Turbidity: Độ đục
# 7.Colour: Màu sắc
# 8.Grade: Chất lượng sữa

from __future__ import division
from unittest import result
from numpy import double
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np


#---------------------------------------------------------------------------------

def predictValue(dataCount,data, dicY, LenY):
    result = {}
    # print("Start")
    for i in dicY:
        P = (double)(dicY.get(i)/LenY)                                  #P(Class)
        # print("\n(",dicY.get(i), "/",LenY, ") *")
        for j in range(len(data)):
            for x in range(len(dataCount)):
                if(j == x):
                    if dataCount[x].get(data[j]) is None:                           #Kiểm tra giá trị chưa tồn tại trong mô hình
                        continue
                    else:
                        # print(data[j], dataCount[x])                            #data[j]: giá trị dự đoán,  dataCount[x]: Bảng tần xuất tại thuộc tính của giá trị
                        # print((dataCount[x].get(data[j])))                        #(dataCount[x].get(data[j]): Tần xuất của giá trị
                        # print((dataCount[x].get(data[j])).get(i))                 #(dataCount[x].get(data[j])).get(i): Tần suất giá trị tại lớp i đang xét
                        if (dataCount[x].get(data[j])).get(i) is None:
                            # print(" * (0/",dicY.get(i),")")
                            P = P * 0
                        else:
                            # print(" * (",(dataCount[x].get(data[j])).get(i),"/",dicY.get(i),")")        
                            P = P * ((dataCount[x].get(data[j])).get(i) / dicY.get(i))                      #P(X/class)
        result[i] = P
    
    maxValue = 0
    resultClass = 0
    for i in dicY:                                                              #Phân lớp
        if result.get(i) > maxValue:
            maxValue = result.get(i)
            resultClass = i
    if(maxValue == 0):                                                      #Không xác định
        resultClass = -1
    return result, resultClass

def countValue(X, properties, Y,):
    data = []
    for propertie in range(properties):             #lặp qua các thuộc tính
        # print("TT: ", propertie)
        dt = []
        for valueX in X:
            dt.append(valueX[propertie])             #Lấy các giá trị của thuộc tính
        dicVL = {}
        for i in set(dt):                               #Lặp qua các giá trị
            dicY = {}
            for j in range(len(dt)):                    
                for k in range(len(Y)):
                    if dt[j] == i and k == j:            #Kiểm tra giá trị đang xét và thuộc tính Y tương ứng với giá trị đang xét, k == j giá trị X tương ứng Y
                        if Y[k] in dicY:                 #Đếm số lần xuất hiện của Y tại thuộc tính đang xét
                            dicY[Y[k]] += 1
                        else:
                            dicY[Y[k]] = 1
            dicVL[i] = dicY                              #Thêm vào từ điển key: THuộc giá trị đang xét, value: Tần suất xuất hiện tại các lớp
        # print(dicVL)             
        data.append(dicVL)
    
    dicY = {}
    for valueY in Y:                                      #Đếm số lần xuất hiện của nhãn thuộc tính phân lớp
        if valueY in dicY:
            dicY[valueY] += 1
        else:
            dicY[valueY] = 1
    return data, dicY


#------------Nhập dữ liệu và xử lí-----------------
DataTrain=pd.read_csv("milk_train.csv")
Xtrain=DataTrain.drop('Grade',axis=1)
YtrainDf=DataTrain["Grade"].values
Ytrain = []
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
##------------------------------Sử dụng thư viện--------------------------------
naiveModel = GaussianNB()
naiveModel.fit(Xtrain,Ytrain)
predictNave=naiveModel.predict(Xtest)

print("\n-------------------Thuật toán Naive_Bayes-----------------------")
print("\nGiá trị dự đoán (Thư viện): ")
precisionLb = round(precision_score(Ytest, predictNave, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionLb,"%\n")

##--------------------------Code thuần--------------------------------

data, dicY = countValue(Xtrain.values, len(Xtrain.values[0]), Ytrain)                #lấy số thuộc tính
# print("\nTần suất xuất hiện:")
# for i in range(len(data)):
#     print("Thuộc tính: ",i, ", Tần suất giá trị: ",data[i])

print("Giá trị dự đoán (Không sử dụng thư viện): ")
dataClassPredict = []
for i in Xtest.values:
    result,resultClass = predictValue(data,i,dicY, len(Ytrain))                                          #Dự đoán
    dataClassPredict.append(resultClass)
# # print(dataClassPredict)
precisionNaive = round(precision_score(Ytest, dataClassPredict, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionNaive,"%\n")
