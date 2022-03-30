from tarfile import XHDTYPE
from unittest import result
import numpy as np
import math

X = [[1,1],[2,1],[4,3],[5,4]]
C = (int)(input("Số tâm = "))

Cdt = {}
Countitem = len(X[0])                               #Lấy số lượng phần từ của 1 mẫu X
for i in range(C):                                  #Khởi tạo tâm
    arr = {}
    arr[i] = X[i]
    Cdt[i] = arr

def Distane(value, Cdt):
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
        print(arr)
        min = arr.get(0)
        for k in arr:   
            resultClus = k                                    
            if min > arr.get(k):
                min = arr.get(k)
                resultClus = k                          #Lấy ra cụm tâm gần nhất
    (Cdt.get(resultClus))[i] = X[i]                     #Thêm giá trị mới vào tâm
    return Cdt

# Cập nhật vị trí trọng tâm
def UpdateClustering():
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

Cdt = Distane(C, Cdt)
print(Cdt)

Cdt = UpdateClustering()
# print(Cdt)


# Cdt = Distane(0)
# print(Cdt)

# Cdt = UpdateClustering()
# print(Cdt)


# Cdt = Distane(0)
# Cdt = UpdateClustering()
