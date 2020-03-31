import numpy as np

from sigmoid import sigmoid

#.....Cost Function.....
def CostFunction(theta,X,y) :
    m=len(y)
    z=np.dot(X,theta)
    hx=sigmoid(z)
    a=np.dot(y.T,np.log(hx))
    b=np.dot((1-y).T,np.log(1-hx))
    J=np.sum(a+b)/(-m)
    return J
