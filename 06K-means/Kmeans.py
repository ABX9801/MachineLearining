# This is the implementation of the K-Means Algorithm;
# This is an Unsupervised Machine Learning Algorithm;
# Objective is to classify the different data points into 2 clusters( Visualization - with different colors)
# I am using My own data(the one I used in logistic Regression)

import numpy as np 
import random
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import math

style.use('fivethirtyeight')

X = [[0,2],[0,3],[1,2],[1,3],[2,3],[2,0],[3,0],[2,1],[3,1],[3,2]]
m = len(X)
for i in range(m):
    plt.scatter(X[i][0],X[i][1],color='k')

def distance(x,y):
    d=0
    for i in range(len(x)):
        d+= (x[i]-y[i])**2
    return math.sqrt(d)

def similarity(x,y):
    t=0
    for i in range(len(x)):
        if(x[i]==y[i]):
            t+=1
    if(t==2):
        r=True
    else:
        r=False
    return r

def avg(list):
    n = len(list)
    t=[0,0]
    for i in range(n):
        t = np.add(t,list[i])
    for i in range(2):
        t[i]=t[i]/n
    return t

def Cost_Function(x_data,c,mu):
    j=0
    for i in range(m):
        j+=(np.linalg.norm(np.array(x_data[i])-np.array(mu[c[i]])))**2
    return j/m

def KMeans(x_data,mu):
    while(True):
        c=[]
        for i in range(m):
            if(distance(x_data[i],mu[0])<distance(x_data[i],mu[1])):
                c.append(0)
            elif(distance(x_data[i],mu[0])>distance(x_data[i],mu[1])):
                c.append(1)
        temp = []
        for k in range(2):
            t=[]
            for i in range(m):
                if(k==c[i]):
                    t.append(x_data[i])
            temp.append(avg(t))
        ct=0
        for j in range(2):
            if(similarity(temp[j],mu[j])):
                ct+=1
        if(ct==2):
            break
        else:
            for i in range(2):
                mu[i] = temp[i]
    
    return c,mu

C,MU = KMeans(X,[[1,2],[0,2]])
# Only Random Initialization is left    

         
for i in range(m):
    if(C[i]==0):
        r = 'b'
    else:
        r = 'r'
    plt.scatter(X[i][0],X[i][1],color=r)
plt.show()
