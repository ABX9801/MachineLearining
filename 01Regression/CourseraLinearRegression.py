#This is The Python implementation of the Linear Regression Algorithm taught by Andrew Ng 
#I am using my own data; I know that it will come by positive slope

#Steps Needed
# 1)Find Hypothesis
# 2)Find Cost Function
# 3)Go for Gradient Descent (We will assume alpha to be 0.01)

#We will also plot the regression using matplotlib 

#NOTE;
# The dataset will only have single feature;


# Importing important libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
from statistics import mean
style.use("fivethirtyeight")




#Getting Data
df = pd.read_csv("C:\\Users\\ABX9801\\Documents\\houses.txt")

X = np.array(df['size'],dtype = np.float64)
Y = np.array(df['cost'],dtype = np.float64)

#X = np.array([1,2,3,4,5,6],dtype = np.float64) 
#Y = np.array([5,4,6,5,6,7],dtype = np.float64)

    
m = int(len(X)) # Number of training Examples

#Scaling DATA
#Scaling of Data is necessary for the imported data;
#Not only for the imported data I personally think Scaling of Data is always necessary;
def scale(x_data):
    ma = max(x_data)
    mi = min(x_data)
    me = mean(x_data)
    for i in range(m):
        x_data[i] = (x_data[i]-me)/(ma-mi)
    
scale(X) #To understand the importance of scaling comment out this part and see what happens

#hypothesis = O1*+O2*X
#Cost Function = J(O1,O2) = (1/2m)*summation(h(xi)-y(i))
def hypothesis(i,theta,x_data):
    x = theta[0] + theta[1]*x_data[i]
    return x

def Cost_Function(theta,x_data,y_data):
    x = 0
    for i in range(m):
        x+= (1/2*m)*((hypothesis(i,theta,x_data)-y_data[i])**2)

    return x

def Cost_Function_Derivative(theta,x_data,y_data,theta_index):
    # dJ/dtheta0 = (1/m)(h(xi)-y(i))
    # dJ/dtheta1 = (1/m)(h(xi)-y(i))*x(i)
    d = 0
    for i in range(m):
        if theta_index==0:
            d += (1/m)*(hypothesis(i,theta,x_data)-y_data[i])
        elif theta_index==1:
            d += (1/m)*(hypothesis(i,theta,x_data)-y_data[i])*x_data[i]
    
    return(d)

#The below is the Gradient Descent algorithm;
#We can say that it is an optimization algorithm;
#The Gradient_Descent functions returns the list theta; which has theta values such that the cost function will be minimum
def Gradient_Descent(theta,alpha,x_data,y_data):
    temp =[0,0]
    while(True):
        cf_old = Cost_Function(theta,x_data,y_data)
        temp[0] = theta[0] - (alpha)*(Cost_Function_Derivative(theta,x_data,y_data,0))
        temp[1] = theta[1] - (alpha)*(Cost_Function_Derivative(theta,x_data,y_data,1))
        cf_new = Cost_Function(temp,x_data,y_data)
        if(cf_new<cf_old):#Simultaneous update
            theta[0] = temp[0]
            theta[1] = temp [1]
        elif(cf_new>cf_old):
            break
        elif(cf_new==cf_old):
            break
    
    return theta

p = Gradient_Descent([5,5],0.01,X,Y) #you can use any theta value it doesn't matter;


for i in range(m):
    plt.scatter(X[i],Y[i],s =20, color = 'k')
#plotting the training data scatter plot

h =[]
for i in range(m):
    h.append(hypothesis(i,p,X))
#ploting the hypothesis line for our graph

plt.plot(X,h,color="green")
plt.show()