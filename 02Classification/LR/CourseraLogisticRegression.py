# This is the Python Implementation of Logistic Regression as taught by Andrew Ng
# I am using my own data 
# I will also plot the decision boundry and also the scatter plot of training data

#Steps Needed;
#1) Calculate Hypothesis
#2) Calculate Cost Function
#3) Calculate Cost Function Gradient
#4) Get Optimal theta using Gradient Descent

#NOTE;
# The data will have only 2 features and will be divided into classes either 0 or 1

import numpy as np
from math import log,exp
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#Importing Data as [[x1,x2,class],[x1,x2,class],[x1,x2,class]]
data = [[0,2,1],[0,3,1],[1,2,1],[1,3,1],[2,3,1],[2,0,0],[3,0,0],[2,1,0],[3,1,0],[3,2,0]]
m = len(data)
X=[]
Y=[]
for i in range(m):
    X.append([1.0,data[i][0],data[i][1]])#Since we can say that the first feature is 1
    Y.append(data[i][2])

#for i in range(m):
#    print(X[i],":",Y[i])

def Hypothesis(theta,x_data,i):
    h = np.dot(theta,x_data[i])
    return 1/(1+exp(-h))

def Cost_Function(theta,x_data,y_data):
    j = 0
    for i in range(m):
        j = j + (-1/m)*(y_data[i]*log(Hypothesis(theta,x_data,i)) + (1-y_data[i])*log(1-Hypothesis(theta,x_data,i)))
    
    return j

def Cost_Function_Derivative(theta,x_data,y_data,t):
    dj = 0
    for i in range(m):
        dj = dj+(1/m)*(Hypothesis(theta,x_data,i)-y_data[i])*x_data[i][t]

    return dj

def Gradient_Descent(theta,x_data,y_data,alpha,iterations):
    for  i in range(iterations):
        t1 = theta[0]-alpha*(Cost_Function_Derivative(theta,x_data,y_data,0))
        t2 = theta[1]-alpha*(Cost_Function_Derivative(theta,x_data,y_data,1))
        t3 = theta[2]-alpha*(Cost_Function_Derivative(theta,x_data,y_data,2))
        theta[0] = t1
        theta[1] = t2
        theta[2] = t3
        # We can also use the same logic as that of Linear Regression to get near Optimal Value of theta ;
        # but here if our learning rate is more precise; the longer it will take to get optimal theta;
        # So we use the theory of itertions;

    return theta

p = Gradient_Descent([1,1,1],X,Y,0.5,10000)
x1 = []
x2 = []
for i in range(m):
    x1.append(X[i][1])
    x2.append(X[i][2])

db =[]
for i in range(m):
    t = -(p[0]+p[1]*x1[i])/p[2]
    db.append(t)

for i in range(m):
    if(Y[i]==0):
        c = 'r'
    else:
        c = 'b'
    plt.scatter(x1[i],x2[i],s=50,color = c)

plt.plot(x1,db,color = 'k')
plt.show()
