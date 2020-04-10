#We will use our made classifier to predict the features :[4,2,1,1,1,2,3,2,1],[4,9,7,6,8,5,4,9,7];
#We will also get the accuracy of our classifier

#NOTE;
# 1- We recreate our data by framing it into dictionary format;
# 2- We split the data ourselves into training and testion set;
# 3- We train the training data and then test the testing data and predict accuracy

import random
import numpy as np 
import pandas as pd 
from collections import Counter


#THE K-Nearest Neighbor classifier function
def KNearestNeighbors(data,predict,k=5):
    distances =[]
    for i in data:
        for j in data[i]:
            Eucledian_distance = np.linalg.norm(np.array(j)-np.array(predict))
            #in the data we have a more than 2-d features ; so for faster calculation of Eucledian distance we use inbuilt ;
            #linear algebra function of numpy
            distances.append([Eucledian_distance,i])#making a list as [[distance,class],[distance,class]]
    vote = []
    for i in sorted(distances)[:k]:
        vote.append(i[1]) #Appending the first 3 classes of the sorted array
    l = Counter(vote).most_common(1)[0][0]   #Counting the votes ; segregating the most common
    #; returning class of most common to l
    return l

#NOTE
#4 for Malignant ;
#2 for Benign ;  

df = pd.read_csv("C:\\Users\\ABX9801\\Documents\\breast-cancer-wisconsin.data")
df.replace('?',-99999,inplace =True)
df.drop(['id'],1,inplace = True)

d = df.astype(float).values.tolist()#converting all string values of data to float and passing them into a list
random.shuffle(d)

test_size = 0.2
X_train = {2 : [], 4 : []}
X_test = {2 : [], 4 : []}
traindata = d[:-int(test_size*len(d))]
testdata = d[-int(test_size*len(d)):]

for i in traindata:
    X_train[i[-1]].append(i[:-1])

for i in testdata:
    X_test[i[-1]].append(i[:-1])

correct = 0
total = 0
for i in X_test:
    for j in X_test[i]:
        temp = KNearestNeighbors(X_test,j,k=3) 
        if(temp==i):
            correct+=1
        total+=1

accuracy = (correct/total)
print(accuracy)

t = KNearestNeighbors(X_train,[4,9,7,6,8,5,4,9,7],k=5) # u can change the above list
print(t)