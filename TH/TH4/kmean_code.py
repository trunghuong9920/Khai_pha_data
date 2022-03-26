import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import precision_score


Data = pd.read_csv("glass.csv")
#print(Data)
Data.head
X= Data.drop('Type', axis=1).values
Y = Data["Type"].values
n = X.shape[0]#chiều rộng mảng
r = X.shape[1]#chiều dài mảng
C = 3
#Khởi tạo tâm
V = np.zeros((C, r))
D = np.zeros((n, C))
Dmin = np.zeros(n)
cum = np.zeros(n)
dem = np.zeros(C)
tong = np.zeros((C, r))
for i in range(C):
    V[i] = X[i]
Maxstep = 100
t = 0
print(V)
while True:
    t = t + 1
    for i in range(n):
        for j in range(C):
            D1 = 0
            for l in range (r):
                D1 = D1+(X[i, l] - V[j, l])*(X[i,l]- V[j, l])
            D[i,j] = math.sqrt(D1)
            # if Dmin[i]<(1/D[i,j]):
            # Dmin[i] =D[i,j]
    Dmin = np.min(D, axis=1)
    for i in range(n):
            for j in range(C):
                if D[i,j] == Dmin[i]:
                    cum[i]=j
    V1 = np.zeros((C, r))
    #Tính lại tâm
    for j in range(C):
        for i in range(n):
            if cum[i] == j:
                dem [j] = dem[j] + 1
                for k in range(r):
                    tong [j, k] = tong[j,k]+X[i, k]
    for j in range(C):
        for k in range(r):
            V1[j, k] = tong[j, k]/dem[j]
    if t > Maxstep:break
    DV = np.zeros(C)
    for i in range(C):
        D1 = 0
        for l in range(r):
            D1 =D1 + (V1[i, l] - V[i, l]) * (V1[i, l] - V[i, l])
            DV[i] = math.sqrt(D1)
    DVmax = np.max(DV)
    if DVmax<0.05: break
    V1 = V1
print(cum)
print(V)
print(t)
