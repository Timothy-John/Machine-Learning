import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.optimize as scio

from sigmoid import sigmoid
from CostFunction import CostFunction
from Gradient import Gradient
from PlotData import PlotData
from DecisionBoundary import DecisionBoundary
from accuracy import accuracy

#.....Read Data.....
data=np.loadtxt('ex2data1.txt',delimiter=",")
X=data[:,0:2]
y=data[:,2]
m=len(y)
theta=np.zeros([3,1])

#.....Plotting Data.....
PlotData(data,'Exam Score 1','Exam Score 2','Admitted','Not Admitted')
plt.show()

#.....Cost Function.....
X=np.hstack((np.ones((m,1)),X))
y=y[:,np.newaxis]                #adding to correctly compute Gradient

J=CostFunction(theta,X,y)    
print("Cost (with Intitial Theta) = ",J)   

#.....Gradient.....
grad=Gradient(X,y,theta)
print("Gradient= \n",grad)

#.....Optimizing using SCIPY.....          #Gradient Descent can be used 
res = scio.minimize(CostFunction, theta, method='TNC', args=(X, y), options={'maxiter': 1000})
theta = res.x
J = res.fun
print("Theta=\n",theta)
print("Cost with Theta=\n",J)

#.....Plot Dicision Boundary.....
DecisionBoundary(data,theta)
plt.show()

#.....Accuracy.....
count=0
for i in range(m):
     p=accuracy(X[i,:],theta)
     if p==y[i]:
         count+=1
print("Accuracy in predicting : ",(count/m)*100) 
 
#.....Probability Prediction.....
print("\nEnter mark1 and mark2 of Student.....")
m1=int(input('Mark 1: '))
m2=int(input('Mark 2: '))
prob=sigmoid(np.dot(np.array([1, m1, m2]),theta))
print("\nChance of Student getting admission: ",prob*100)
