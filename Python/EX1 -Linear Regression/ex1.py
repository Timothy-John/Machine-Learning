import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

from ComputeCost import ComputeCost
from GradientDescent import GradientDescent

#.....Loading Data.....
data=np.loadtxt('ex1data1.txt',delimiter=",")   #pandas ERROR!!
x=data[:,0]
y=data[:,1]
m=len(y)

#.....Plotting.....
plt.scatter(x,y)
plt.title('PLOT GRAPH')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in Rs.10,000s')
plt.show()
#.....Cost Function.....
X=x[:,np.newaxis]                        #increasing dimension
y=y[:,np.newaxis]                        #increasing dimension
theta = np.zeros([2,1])
X = np.hstack((np.ones([m,1]), X))       #addind intercept term
iterations = 1500
alpha = 0.01

J=ComputeCost(X,y,theta)
print("Cost=",J)

#.....Gradient Descent.....
theta,J_history=GradientDescent(X,y,theta,iterations,alpha)
print("Theta= \n",theta)

#.....Optimum Cost.....
J=ComputeCost(X,y,theta)
print("Optimised Cost= ",J)

#.....Plotting Best Fit Graph.....
plt.scatter(x,y)
plt.title('BEST FIT GRAPH')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in Rs.10,000s')
plt.plot(x,np.dot(X,theta),color='green')
plt.show()

#.....Plotting Convergence Graph.....
its=np.zeros([iterations,1])
for i in range (iterations):
    its[i]=i

plt.plot(its,J_history)
plt.title('CONVERGENCE GRAPH')
plt.xlabel('No: of Iterations')
plt.ylabel('Cost Function')
plt.show()
