import numpy as np

from ComputeCost import ComputeCost

#.....Gradient Descent.....
def GradientDescent(X,y,theta,iterations,alpha):
    J_history=np.zeros([iterations,1])
    m=len(y)
    for i in range (iterations):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta=theta-(alpha/m)*temp

        J_history[i]=ComputeCost(X,y,theta)
    return theta,J_history