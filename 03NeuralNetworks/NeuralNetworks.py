# This is the Python implementation of Neural Networks taught by Andrew Ng
# We are using the standard iris dataset to perform multiclass classification
# Iris-setosa = [1,0,0]
# Iris-versicolor = [0,1,0]
# Iris-virginica = [0,0,1]
import numpy as np
import pandas as pd
import random
from math import log,exp

data = pd.read_csv("C:\\Users\\ABX9801\\Documents\\iris.data")

# Changing string classes into vectors as decided above
X  = np.array(data.drop(['class'],1))
m  = len(X)
Yi = np.array(data['class'])
Y = []
for i in range(m):
    if(Yi[i]=='Iris-setosa'):
        Y.append([1,0,0])
    elif(Yi[i]=='Iris-versicolor'):
        Y.append([0,1,0])
    elif(Yi[i]=='Iris-virginica'):
        Y.append([0,0,1])
# Making the dataset with x and y values in list as;
#T = [[[x1,x2,x3,x4],[class]],[[x1,x2,x3,x4],[class]],[[x1,x2,x3,x4],[class]]]
T = []
for i in range(m):
    T.append([X[i],Y[i]])
# Shuffling the arranged data and seperating X and Y data
random.shuffle(T)
X = []
Y = []
for i in range(m):
    X.append(T[i][0])
    Y.append(T[i][1])
# Taking 80% data for training
train_size =0.8

x_train = X[:(int(train_size*m))]
y_train = Y[:(int(train_size*m))]
mt = len(x_train)
x_test = X[(int(train_size*m)):]
y_test = Y[(int(train_size*m)):]


# Now we have turned our classes into list;
# Now we select an architecture input features = 4;
# output features = 3 , hidden layers = 1, hidden_feature = 5(our call)

# hidden_features = 5 ; input_features = 4 
#theta1 = 5x5 matrix
t1 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

#hidden_features =5 ; output_features = 3
#theta2 = 3x6 matrix
t2 = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]

#initializing with 0 does not work; so let us randomly initialize the matrix; let epsilon = 0.001
epsilon = 0.001
for i in range(5):
    for j in range(5):
        t1[i][j] = random.uniform(-epsilon,epsilon)

for i in range(3):
    for j in range(6):
        t2[i][j] = random.uniform(-epsilon,epsilon)

# Hypothesis Uses Feed Forward propagation with adding Bias unit to all layers leaving the output layer
def hypothesis(theta1,theta2,x_data,i):
    a2 = [1]
    a3 = []
    temp =[1]
    for j in range(4):
        temp.append(x_data[i][j])
    for j in range(5):
        z1 = np.dot(temp,theta1[j])
        g1 = (1/(1+exp(-z1)))
        a2.append(g1)
    for j in range(3):
        z2 = np.dot(a2,theta2[j])
        g2 = (1/(1+exp(-z2)))
        a3.append(g2)
    return a3 , a2 , temp
##NOTE that mt =  no. of training examples here coz m =  total length of dataset

# The Cost Function For the Data; We have to minimize it Further
def Cost_Function(x_data,y_data,theta1,theta2):
    j = 0
    for i in range(mt):
        h , a = hypothesis(theta1,theta2,x_data,i)
        for k in range(3):
            j += ((y_data[i][k]*log(h[k]) + (1-y_data[i][k])*log(1-h[k])))
    return j*(-1/mt)

# Transpose function is needed for matrix multiplication
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

# Backpropagation algorithm to get dJ/dtheta
#first we get delta(error) values for all layers;
# then we fill up the del matrices 
def Backprop(x_data,y_data,theta1,theta2):
    del1 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    del2 = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    for i in range(mt):
        a3 , a2 , x = hypothesis(theta1,theta2,x_data,i)
        delta3 = np.subtract(a3,y_data[i])
        tt2 =[]
        for j in range(3):
            tt2.append(theta2[j][1:])
        # Now tt2 is new theta2 with removed bias unit; 3x5 matrix
        # Now we got to get the transpose of the matrix; to get 5x3
        tt2 = Transpose(tt2)
        f = []
        for j in range(len(tt2)):
            f.append(np.dot(tt2[j],delta3))
        # f is a 5x1 matrix now
        af = a2[1:]
        ones =[]
        for j in range(len(af)):
            ones.append(1)
        afi = np.subtract(ones,af)
        for j in range(len(af)):
            af[j] = af[j]*afi[j]
        #af is a 5x1 matrix 
        delta2 = []
        for j in range(len(af)):
            delta2.append(f[j]*af[j])
        # Now we got delta1 and delta2 values
        for j in range(5):
            for k in range(5):
                del1[j][k] += x[k]*delta2[j]
        # we got del1 now we go for del2
        for j in range(3):
            for k in range(6):
                del2[j][k] += a2[k]*delta3[j]
    return del1,del2

#Using Gradient Descent to get the optimal theta according to us
def Gradient_Descent(theta1,theta2,x_data,y_data,iterations,alpha):
    for k in range(iterations):
        #for del1
        d1,d2 = Backprop(x_data,y_data,theta1,theta2)
        for i in range(5):
            for j in range(5):
                theta1[i][j] = theta1[i][j] - alpha*(1/mt)*d1[i][j]
        
        for i in range(3):
            for j in range(6):
                theta2[i][j] =theta2[i][j] - alpha*(1/mt)*d2[i][j]
    
    return theta1,theta2

tf1,tf2 = Gradient_Descent(t1,t2,x_train,y_train,10000,1.2)

# The bellow code checks our accuracy on the training dataset
correct = 0
for i in range(len(x_test)):
    op,hi,ip = hypothesis(tf1,tf2,x_test,i)
    for j in range(3):
        if(op[j]<0.5):
            op[j]=0
        else:
            op[j]=1
    c =0
    for j in range(3):
        if(op[j]==y_test[i][j]):
            c+=1
    if(c==3):
        correct+=1

print("Accuracy : ",(correct/len(x_test)*100))