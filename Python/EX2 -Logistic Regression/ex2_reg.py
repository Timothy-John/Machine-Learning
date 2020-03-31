import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scio

from PlotData import PlotData
from CostFunctionReg import CostFunctionReg
from DecisionBoundary import DecisionBoundary_reg
from Gradient import Gradient
from MapFeature import MapFeature
from accuracy import accuracy

#.....Load Data.....
data=np.loadtxt('ex2data2.txt',delimiter=",")
X=data[:,0:2]
y=data[:,2]
lam=1            #.....Change this value

#.....Plotting Data.....
PlotData(data,'Microchip Test 1','Microchip Test 1','Accepted','Not Accepted')
plt.show()

#.....Map Featuring with SKLEARN.....
X=MapFeature(X)
m=X.shape[0]

#.....Reguralized Cost Function.....
X=np.hstack((np.ones((m,1)),X))
y=y[:,np.newaxis]         #adding to correctly compute Gradient
theta=np.zeros([X.shape[1],1])

J=CostFunctionReg(theta,X,y,lam)
print("Cost (with Initial Thet) = ",J)

#.....Gradient.....
grad=Gradient(X,y,theta)
print("First 5 Gradients=\n",grad[0:5,:])

#.....Optimization using SCIPY.....           Gradient Descent can be used
print("........Training with Data Set")
res=scio.minimize(CostFunctionReg,theta,method='CG',args=(X,y,lam),options={'maxiter':1000})
theta=res.x
J=res.fun
print("\nTheta=\n",theta)
print("\nCost with Theta= ",J)

#.....Decision Boundary.....
DecisionBoundary_reg(data,theta)
plt.show()

#.....Accuracy.....
count=0
for i in range(m):
     p=accuracy(X[i,:],theta)
     if p==y[i]:
         count+=1
print("\nTrain Accuracy: ",(count/m)*100) 
