# This is the implementation of Principle Control Analysis;
# We do this for Compression of Data; and its easy Visualization;
# We will be using Linear Algebra Libraries in numpy to use implement it;
# Most probably the data we will use is the Iris dataset;(n =4 features)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
from statistics import mean


style.use('fivethirtyeight')

# we need the below function to facilitate our matrix multiplication
def Transpose(l):
    l1 = len(l)
    l2 = len(l[0])
    temp =[]
    for i in range(l2):
        t =[]
        for j in range(l1):
            t.append(l[j][i])
        temp.append(t)

    return temp

data = pd.read_csv("C:\\Users\\ABX9801\\Documents\\iris.data")

X  = np.array(data.drop(['class'],1))
m  = len(X)
Y = np.array(data['class'])
X = np.array(Transpose(X))

# Scalingof data is needed
def scale(x_data):
    ma = max(x_data)
    mi = min(x_data)
    me = mean(x_data)
    for i in range(m):
        x_data[i] = (x_data[i]-me)/(ma-mi)

for i in range(4):
    scale(X[i])
X = Transpose(X)

# Now we have donr the scaling
# We will now implement PCA and get the Z vector

# This is doing PCA for ith training example
def PCA(x_data,i):
    for j in range(4):#we know that we have four features;
        x_data[i][j]=[x_data[i][j]]
    sigma = np.matmul(np.array(x_data[i]),np.array(Transpose(x_data[i])))
    U,S,V = np.linalg.svd(sigma)
    Uf=[]
    for i in range(len(U)):
        t=[]
        for j in range(2):
            t.append(U[i][j])
        Uf.append(t)
    z = np.matmul(np.array(Transpose(Uf)),np.array(x_data[i]))
    return z

Z=[]
for i in range(m):
    Z.append(PCA(X,i))

for i in range(m):
    if(Y[i]=='Iris-setosa'):
        c='r'
    elif(Y[i]=='Iris-versicolor'):
        c='b'
    elif(Y[i]=='Iris-virginica'):
        c='g'
    plt.scatter(Z[i][0],Z[i][1],color=c)

plt.show()
    
