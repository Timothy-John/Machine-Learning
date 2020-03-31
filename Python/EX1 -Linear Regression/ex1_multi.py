import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

from FeatureNormalize import FeatureNormalize
from ComputeCost import ComputeCost
from GradientDescent import GradientDescent

#.....Read Data.....
     #Cannot use pandas to read txt !!ERROR
data=np.loadtxt('ex1data2.txt',delimiter=",")
X=data[:,0:2]
y=data[:,2]
m=len(y)
theta=np.zeros([3,1])
iterations=1500
alpha=0.01

#.....Normalizing Feature.....
X=FeatureNormalize(X)
#X=X[:,np.newaxis]     !!ERROR
#y=y[:,np.newaxis]     !!ERROR
X=np.hstack((np.ones((m,1)),X))      #adding intercept term

#.....Gradient Descent.....
theta,J_history=GradientDescent(X,y,theta,iterations,alpha)
#print("Theta= ",theta) 

#.....Plotting Convergence graph.....
its=np.zeros([iterations,1])
for i in range(iterations):
    its[i]=i

plt.plot(its,J_history)
plt.title('CONVERGENCE GRAPH')
plt.xlabel('No: of Iterations')
plt.ylabel('Cost Function')
plt.show()
