import scipy.io as scio   #only to import .mat file
import numpy as np
import matplotlib.pyplot as plt
import random

from DisplayData import DisplayData
from DisplayData_nn import DisplayData_nn
from CostFunction import CostFunction
from OneVsAll import OneVsAll
from accuracy import accuracy

#.....Load Data.....
print("\nLoading and Visualizing Data.......")
data=scio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
m,n=X.shape
nums=10    #for numbers from 0 to 9
theta=np.zeros([n+1,1])
lam=1      #change regularization parameter

#.....Visualizing data.....
DisplayData(X)
plt.show()

#.....Cost Function.....
X=np.hstack((np.ones([m,1]),X))
J=CostFunction(theta,X,y,lam)
print("\nTesting Cost Function: SUCCESS....")

#.....Training using SCIPY.....
theta_trained=OneVsAll(theta,X,y,lam,nums)  # Gradient Descent can also be used

#.....Accuracy.....
count=0
print("Accuracy of Algorithm : ")
for i in range(0,5000):
     p=accuracy(theta_trained,X[[i],:])
     if p==y[i]:
         count+=1
print((count/5000)*100)      

#.....Prediction of Labels.....   
print("\nPredicting values in Dataset.....")
for i in range(0,m):
    pos=random.randint(0,m)
    DisplayData_nn(X[pos,:])
    pred=accuracy(theta_trained,X[[pos],:])
    print("Predicted Value: ",pred)
    ans=input("Press q to Quit.....")
    if ans=='q':
        break
