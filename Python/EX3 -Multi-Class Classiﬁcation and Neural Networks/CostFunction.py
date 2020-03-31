import numpy as np

from sigmoid import sigmoid

#.....Cost Function.....
def CostFunction(theta,X,y,lam):
    m=len(y)
    hx=sigmoid(np.dot(X,theta))
    a=np.dot(y.T,np.log(hx))
    b=np.dot((1-y).T,np.log(1-hx))
    J=(-(a+b)/m)+((lam*np.dot(theta.T,theta))/(2*m))
    return J
    