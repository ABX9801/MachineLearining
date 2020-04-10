# This is an Implementation of Anomaly Detection Algorithm;
# Its a simple Gaussian Distribution example;
# I will choose my own value of epsilon;

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from scipy.io import loadmat 
import random
import math
from math import sqrt,exp

style.use('fivethirtyeight')

data = loadmat("C:\\Users\\ABX9801\\Documents\\anomaly.mat")
X = data['X']  
m = len(X)

mu=[0,0]
for j in range(2):
    for i in range(m):
        mu[j]+=X[i][j]/m

sigma=[0,0]
for j in range(2):
    for i in range(m):
        sigma[j] += ((X[i][j]-mu[j])**2)/m
for i in range(2):
    sigma[i] = math.sqrt(sigma[i])

pi = math.pi

def Probability(x_data):
    t=1
    for i in range(len(x_data)):
        t = t*((1/(sqrt(2*pi)*sigma[i]))*exp(-((x_data[i]-mu[i])**2/(2*(sigma[i])**2))))
    return t

epsilon = 0.015
