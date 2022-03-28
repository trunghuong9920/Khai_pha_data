from unittest import result
from numpy import double
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

#---------------------------------------------------------------------------------

def predictValue(dataCount,data, Y):
    dicY = {}
    for valueY in Y:
        if valueY in dicY:
            dicY[valueY] += 1
        else:
            dicY[valueY] = 1
    result = {}
    # print("Start")
    for i in dicY:
        P = (double)(dicY.get(i)/len(Y))                                  #P(Class)
        # print("\n(",dicY.get(i), "/",len(Y), ") *")
        for j in range(len(data)):
            for x in range(len(dataCount)):
                if(j == x):
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
    if(maxValue == 0):                                                      #K xác định
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
    return data


#------------Nhập dữ liệu và xử lí-----------------
Data=pd.read_csv("milk_train.csv")
Xtrain=Data.drop('Grade',axis=1)
Ytrain=Data["Grade"].values

Data=pd.read_csv("milk_test.csv")
Xtest=Data.drop('Grade',axis=1)
Ytest=Data["Grade"].values

print("Dữ liệu train: ",len(Xtrain))
print("Dữ liệu test: ",len(Xtest))

#------------------------------NAIVE_BAYESIAN
##------------------------------Sử dụng thư viện--------------------------------
naiveModel = GaussianNB()
naiveModel.fit(Xtrain,Ytrain)
predictNave=naiveModel.predict(Xtest)

print("\n-------------------Thuật toán Naive_Bayesian-----------------------")
print("\nGiá trị dự đoán (Thư viện): ")
# print(predictNave)
precision = round(precision_score(Ytest, predictNave, average='micro') * 100,2)
print("Độ chính xác precision : ", precision,"%\n")


##--------------------------Code thuần--------------------------------

data = countValue(Xtrain.values, len(Xtrain.values[0]), Ytrain)                #lấy số thuộc tính
# print("\nTần suất xuất hiện:")
# for i in range(len(data)):
#     print("Thuộc tính: ",i, ", Tần suất giá trị: ",data[i])

print("Giá trị dự đoán (Code thuần): ")
dataClassPredict = []
for i in Xtest.values:
    result,resultClass = predictValue(data,i,Ytrain)                                          #Dự đoán
    dataClassPredict.append(resultClass)
# # print(dataClassPredict)
precision = round(precision_score(Ytest, dataClassPredict, average='micro') * 100,2)
print("Độ chính xác precision : ", precision,"%\n")



#-------------------------ID3-------------------------
id3Model=tree.DecisionTreeClassifier()
id3Model.fit(Xtrain,Ytrain)
predictId3=id3Model.predict(Xtest)

print("\n-------------------Thuật toán ID3----------------------")
precision = round(precision_score(Ytest, predictId3, average='micro') * 100,2)
print("Độ chính xác precision : ", precision,"%\n")