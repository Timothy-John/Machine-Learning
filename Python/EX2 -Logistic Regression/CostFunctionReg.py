import numpy as np

from sigmoid import sigmoid

#.....Regularized Cost Function.....
def CostFunctionReg(theta,X,y,lam):
    m=len(y)
    z=np.dot(X,theta)
    hx=sigmoid(z)
    a=np.dot(y.T,np.log(hx))
    b=np.dot((1-y).T,np.log(1-hx))
    J=(np.sum(a+b)/(-m))+(pow(lam,2)*np.dot(theta.T,theta))/(2*m)
    return J
