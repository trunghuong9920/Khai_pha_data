import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def ClusterNumber():
    C = []
    x = int(input("Nhập x= "))
    y = int(input("Nhập y= "))
    C.append(x)
    C.append(y)
    return C
#Hàm tính khoảng cách đến tâm cụm
def CalculateDistance(X,y,C):
    N = []
    for i in range(len(X)):
        N.append(math.sqrt(math.pow(X[i]-C[0],2) + math.pow(y[i]-C[1],2)))
    return N

#Hàm phân nhóm 2
def Subgroups2(N1,N2,X,y):
    ResulN1new = []
    ResulN2new = []
    for i in range(len(N1)):
        if(N1[i] < N2[i]):
            ResulN1new.append([X[i],y[i]])
        else:
            ResulN2new.append([X[i],y[i]])
    return ResulN1new,ResulN2new
#Hàm phân nhóm 3
def Subgroups3(N1,N2,N3,X,y):
    ResulN1new = []
    ResulN2new = []
    ResulN3new = []
    for i in range(len(N1)):
        if(N1[i] == min(N1[i],N2[i],N3[i])):
            ResulN1new.append([X[i],y[i]])
        elif(N2[i] == min(N1[i],N2[i],N3[i])):
            ResulN2new.append([X[i],y[i]])
        elif(N3[i] == min(N1[i],N2[i],N3[i])):
            ResulN3new.append([X[i],y[i]])
    return ResulN1new,ResulN2new,ResulN3new
#Hàm tính trung bình cộng
def Averange(newCx,newCy, ResulNnew):
    for i in ResulNnew:
        newCx = newCx+i[0]
        newCy = newCy+i[1]
    newCx = newCx/len(ResulNnew)
    newCy = newCy/len(ResulNnew) 
    return [newCx,newCy]

#Hàm kiểm tra Việc phân 2 nhóm
def checkResuld2(N1, N2,X,y):
    ResulN1new,ResulN2new = Subgroups2(N1,N2,X,y)
    ResulN1old =[]
    ResulN2old =[]
    newCx = 0
    newCy= 0 
    i = 1
    while ResulN1new != ResulN1old or ResulN2new != ResulN2old:
        i = i+1
        ResulN1old = ResulN1new
        ResulN2old = ResulN2new
        C1=Averange(newCx,newCy,ResulN1new)
        C2 = Averange(newCx,newCy,ResulN2new)
        N1 = CalculateDistance(X,y,C1)
        N2 = CalculateDistance(X,y,C2)   
        ResulN1new,ResulN2new = Subgroups2(N1,N2,X,y)
    return ResulN1new, ResulN2new,i,C1,C2

#Hàm kiểm tra Việc phân 3 nhóm
def checkResuld3(N1, N2,N3,X,y):
    ResulN1new,ResulN2new,ResulN3new= Subgroups3(N1,N2,N3,X,y)

    ResulN1old =[]
    ResulN2old =[]
    ResulN3old =[]
    newCx = 0
    newCy= 0 
    i = 1
    while ResulN1new != ResulN1old or ResulN2new != ResulN2old or ResulN3new != ResulN3old:
        i = i+1
        ResulN1old = ResulN1new
        ResulN2old = ResulN2new
        ResulN3old = ResulN3new
        C1 = Averange(newCx,newCy,ResulN1new)
        C2 = Averange(newCx,newCy,ResulN2new)
        C3 = Averange(newCx,newCy,ResulN3new)
        N1 = CalculateDistance(X,y,C1)
        N2 = CalculateDistance(X,y,C2)   
        N3 = CalculateDistance(X,y,C3)   
        ResulN1new,ResulN2new,ResulN3new= Subgroups3(N1,N2,N3,X,y)
    return ResulN1old,ResulN2new,ResulN3new,i,C1,C2,C3

#Hàm trả kết quả
def Result(n,X,y):
    #Phân 2 cụm
    if n==2:
        for i in range(n):
            print("Nhập Tâm: ",i+1)
            C[i] = ClusterNumber()          #Nhập tâm cụm ban đầu
            N[i] = CalculateDistance(X,y,C[i])          #Đưa ra mảng chứa khoảng cách từ điểm đến tâm cụm
        Group1, Group2,i,C1,C2= checkResuld2(N[0],N[1],X,y)                 #Hàm phân nhóm
        return Group1, Group2,i,C1,C2
    #Phân 3 cụm
    if n==3:
        for i in range(n):
            print("Nhập Tâm: ",i+1)
            C[i] = ClusterNumber()
            N[i] = CalculateDistance(X,y,C[i])
        Group1, Group2, Group3,i,C1,C2,C3= checkResuld3(N[0],N[1],N[2],X,y)
        return Group1, Group2, Group3,i,C1,C2,C3

    
#------------------------------------------------------
#-------------------------get data-------------------


# X= [1,1,2,3,4,5]
# y = [4,6,6,8,3,2]

# X= [2,2,1,5,7,6,8,4]
# y = [8,5,2,8,3,4,4,7]

# X = [2,2,8,5,7,6,1,4]
# y = [10,5,4,8,5,4,2,9]

X = [1,2,4,5]
y = [1,1,3,4]

# data = pd.read_csv("diabetes.csv")
# data.head

# X = data["preg"].values
# y = data["plas"].values

C = {}
N = {}
n = int(input("Số cụm cần chia: "))
if n== 2:
    Group1, Group2,i,C1,C2 = Result(n,X,y)
    print("\n-----------------------------------------\n\n",
    "\n\nNhóm 1:",Group1,"\n\nNhóm 2:",Group2,"\n\nTâm cụm 1= ",C1,'\nTâm cụm 2= ',C2)
    print("\nSố lần lặp: ", i)
    
if n== 3:
    Group1, Group2, Group3,i,C1,C2,C3 = Result(n,X,y)
    print("\n-----------------------\n\n",
    "\n\nNhóm 1:",Group1,"\n\nNhóm 2:",Group2,"\nNhóm 3: ",Group3,'\n\nTâm cụm 1= ',C1,'\nTâm cụm 2= ',C2,'\nTâm cụm 3= ',C3)
    print("\nSố lần lặp: ", i)
    

