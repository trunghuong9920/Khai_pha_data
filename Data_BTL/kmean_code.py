from filecmp import cmp
from re import L
from tarfile import XHDTYPE
from unittest import result
import pandas as pd
import copy
import numpy as np

import math

# X = [[1,1],[2,1],[4,3],[5,4]]
# X = [[1,4],[2,6],[1,6],[3,8],[4,3],[5,2]]
# X = [[2,10,8],[2,5,6],[8,4,3],[5,8,9],[7,5,1],[6,4,8],[1,2,5],[4,9,3]]
DataTrain=pd.read_csv("milk_train.csv")
X=DataTrain.drop('Grade',axis=1).values

C = (int)(input("Số tâm = "))

Cdt = {}
Countitem = len(X[0])                               #Lấy số lượng phần từ của 1 mẫu X
for i in range(C):                                  #Khởi tạo tâm
    arr = {}
    arr[i] = X[i]
    Cdt[i] = arr

def Distane(value, Cdt, Cdt1):
    for i in range(value,len(X)):                           #Duyệt qua từng đối tượng trừ tâm
        arr = {}
        for j in Cdt:                                   #duyệt qua tâm
            d = 0
            for x in Cdt.get(j):
                ixJ = 0
                while(ixJ < Countitem):                                     #Lấy giá trị và tính A0 - dc0, A1 - dc1, ....
                    # print((X[i])[ixJ] ,"-",((Cdt.get(j)).get(x))[ixJ] )
                    d += math.pow((X[i])[ixJ] - ((Cdt.get(j)).get(x))[ixJ] , 2)
                    ixJ += 1
                d = math.sqrt(d)
                arr[j] = d                                                   #Thêm giá trị vừa tính được dic
        # print("Khoảng cách= ",arr)
        min = arr.get(0)
        for key in arr.keys():
            resultClus = key
            break
        for k in arr:   
            if min > arr.get(k):
                min = arr.get(k)
                resultClus = k                          #Lấy ra cụm tâm gần nhất
        (Cdt1.get(resultClus))[i] = X[i]                     #Thêm giá trị mới vào tâm
    return Cdt1

# Cập nhật vị trí trọng tâm
def UpdateClustering(Cdt):
    for i in Cdt:                                           #Duyệt qua từng tâm
        tb = len(Cdt.get(i))
        arr = []
        nb = Countitem
        dic = {}
        for k in range(nb):                                 #Lấy phần tử thứ i của tâm       
            d = 0
            for j in Cdt.get(i):    
                # print((Cdt.get(i).get(j))[k])
                d += (Cdt.get(i).get(j))[k]                #Tổng các phần tử thứ i của tâm
            arr.append(d/tb)
        dic[-1] = arr
        Cdt[i] = dic                                        #Cập nhật lại tâm
    return Cdt

def check(dicA,dicB):
    if dicA==dicB:
        return True
    else:
        return False

print("Tâm khởi tạo= ",Cdt)
Cdt1 = copy.deepcopy(Cdt);
oldCluster = Distane(C, Cdt,Cdt1)                           #Phân chia cụm, được tâm oldCluster
print("Tâm cũ",oldCluster)
Cdt = UpdateClustering(copy.deepcopy(oldCluster))           #Tính lại tâm cụm
# print("Tâm mới update",Cdt)


newCluster = copy.deepcopy(Cdt);
for i in newCluster:                                      #Làm rỗng dic chứa tâm mới
    (newCluster.get(i)).clear()
newCluster = Distane(0,Cdt,newCluster)                    #Phân chia cụm tâm mới, được tâm newOldCluster
print("Tâm mới",newCluster)
dem = 1
if check(oldCluster, newCluster) == True:                  #Kiểm tra 2 tâm
    print("Kết quả= ",newCluster)
    print("Số lần lặp= ",dem)
else:                   
    checkDic = False
    while(checkDic == False):
        dem += 1
        oldCluster = copy.deepcopy(newCluster)
        Cdt = UpdateClustering(newCluster)

        newCluster = copy.deepcopy(Cdt);
        for i in newCluster:                                      #Làm rỗng dic chứ tâm mới
            (newCluster.get(i)).clear()
        newCluster = Distane(0,Cdt,newCluster)
        # print("Tâm mới",newCluster)
        if check(oldCluster, newCluster) == True:
            checkDic = True
            print("Kết quả= ",newCluster)
            print("Số lần lặp= ",dem)