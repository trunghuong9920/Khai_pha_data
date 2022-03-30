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
    return data


#------------Nhập dữ liệu và xử lí-----------------
DataTrain=pd.read_csv("milk_train.csv")
Xtrain=DataTrain.drop('Grade',axis=1)

DataTrain = DataTrain.replace(1, 2)
DataTrain = DataTrain.replace(0.5, 1)
Ytrain=DataTrain["Grade"].values

DataTest=pd.read_csv("milk_test.csv")
Xtest=DataTest.drop('Grade',axis=1)

DataTest = DataTest.replace(1, 2)
DataTest = DataTest.replace(0.5, 1)
Ytest=DataTest["Grade"].values

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

data = countValue(Xtrain.values, len(Xtrain.values[0]), Ytrain)                #lấy số thuộc tính
# print("\nTần suất xuất hiện:")
# for i in range(len(data)):
#     print("Thuộc tính: ",i, ", Tần suất giá trị: ",data[i])

print("Giá trị dự đoán (Không sử dụng thư viện): ")
dataClassPredict = []
for i in Xtest.values:
    result,resultClass = predictValue(data,i,Ytrain)                                          #Dự đoán
    dataClassPredict.append(resultClass)
# # print(dataClassPredict)
precisionNaive = round(precision_score(Ytest, dataClassPredict, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionNaive,"%\n")


#-------------------------Kmeans-------------------------
modelKmean = KMeans(n_clusters=3, random_state=25).fit(Xtrain)
predictKmean = modelKmean.predict(Xtest)
print("\n-------------------Thuật toán K-means----------------------")
precisionKmeans = round(precision_score(Ytest, predictKmean, average='micro') * 100,2)
print("Độ chính xác precision : ", precisionKmeans,"%\n")


#-------------------------Biểu đồ------------------------------------
division = ["Naive-Bayes (Thư viện)", "Naive-Bayes (Không dùng Thư viện)", "K-means"]
resultTrue = [precisionLb, precisionNaive, precisionKmeans]
resultFalse = [100-precisionLb, 100-precisionNaive, 100-precisionKmeans]
index = np.arange(3)
width = 0.40

plt.bar(index, resultTrue, width, color="blue", label= "Tỉ lệ dự đoán đúng")
plt.bar(index, resultFalse, width, color="red", label= "Tỉ lệ dự đoán sai", bottom=resultTrue)

plt.title("So sánh tỉ lệ dự đoán của 2 thuật toán")
plt.xlabel("Thuật toán")
plt.ylabel("Thuật toán")

plt.xticks(index, division)

plt.legend(loc = 'best')
plt.show()


# ------------------------------UI----------------------
# -------predic----------
def doan():
    global t1
    global t2
    global t3
    global t4
    global t5
    global t6
    global t7

    t1 = float(entry1.get())
    t2 = float(entry2.get())
    t3 = float(var1.get())
    t4 = float(var2.get())
    t5 = float(var3.get())
    t6 = float(var4.get())
    t7 = float(entry3.get())

    
    Xip = np.array([[t1,t2,t3,t4,t5,t6,t7]])
    Xdf = [t1,t2,t3,t4,t5,t6,t7]

    result1 = naiveModel.predict(Xip)
    result2 = modelKmean.predict(Xip)
    resulta,resultClassa = predictValue(data,Xdf,Ytrain)                                          #Dự đoán

    if result1[0] == 0:
        resultEndNVLB = "Kém"
    if result1[0] == 1:
        resultEndNVLB = "Trung bình"
    if result1[0] == 2:
        resultEndNVLB = "Tốt"

    if result2[0] == 0:
        resultEndid3 = "Kém"
    if result2[0] == 1:
        resultEndid3 = "Trung bình"
    if result2[0] == 2:
        resultEndid3 = "Tốt"

    if resultClassa == -1:
        resultE = "Không xác định"
    elif resultClassa == 0:
        resultE = "Kém"
    elif resultClassa == 1:
        resultE = "Trung bình"
    else:
        resultE = "Tốt"


    label_show.set(resultEndNVLB)
    label_show1.set(resultE)
    label_show2.set(resultEndid3)


root =Tk()
root.option_add("*Font","TimeNewRoman 14")
label_show=StringVar()
label_show1=StringVar()
label_show2=StringVar()
label_show3=StringVar()
label_show4=StringVar()
label_show5=StringVar()
var1 = StringVar(value=0)
var2 = StringVar(value=0)
var3 = StringVar(value=0)
var4 = StringVar(value=0)
ph = StringVar()
temperature = StringVar()
color = StringVar()

 
Label (root, text="Hãy nhập thông tin (nếu thỏa mãn các điều kiện tối ưu vui lòng tích vào ô)").grid(row=0,columnspan=2)

Label (root, text="PH").grid(row=1,column=0,padx=10,pady=10,sticky = W)
entry1 = Entry(root,text="20",textvariable = ph)
entry1.grid(row=1,column=1,padx=10)

Label (root, text="Temprature").grid(row=2,column=0,padx=10,pady=10,sticky = W)
entry2 = Entry(root,text="20",textvariable = temperature)
entry2.grid(row=2,column=1,padx=10)

Label (root, text="Taster").grid(row=3,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var1, onvalue=1, offvalue=0).grid(row=3,column=1,padx=10,pady=10,sticky = W)

Label (root, text="Odor").grid(row=4,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var2, onvalue=1, offvalue=0).grid(row=4,column=1,padx=10,pady=10,sticky = W)

Label (root, text="Fat").grid(row=5,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var3, onvalue=1, offvalue=0).grid(row=5,column=1,padx=10,pady=10,sticky = W)

Label (root, text="Turbidity").grid(row=6,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var4, onvalue=1, offvalue=0).grid(row=6,column=1,padx=10,pady=10,sticky = W)

Label (root, text="Colour").grid(row=7,column=0,padx=10,pady=10,sticky = W)
entry3 = Entry(root,text="20",textvariable = color)
entry3.grid(row=7,column=1,padx=10)

Button (root, text="Kết quả dự đoán", command=doan,bg= 'cyan').grid(row=8,columnspan=2,padx=10,pady=10,sticky = E)

Label (root, text="Naive_Bayes (Thư viện):").grid(row=9,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show).grid(row=9,column=2,padx=10,pady=10,sticky = E)

Label (root, text="Naive_Bayes (Không sử dụng thư viện):").grid(row=10,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show1).grid(row=10,column=2,padx=10,pady=10,sticky = E)

Label (root, text="K-means:").grid(row=11,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show2).grid(row=11,column=2,padx=10,pady=10,sticky = E)

Label (root, text="Tỉ lệ dự đoán Naive_Bayes (Thư viện):").grid(row=12,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show3).grid(row=12,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Tỉ lệ dự đoán Naive_Bayes (Không sử dụng thư viện):").grid(row=13,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show4).grid(row=13,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Tỉ lệ dự đoán K-means (Thư viện):").grid(row=14,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show5).grid(row=14,column=2,padx=10,pady=10,sticky = E)

label_show3.set(precisionLb)
label_show4.set(precisionNaive)
label_show5.set(precisionKmeans)

root.mainloop()
