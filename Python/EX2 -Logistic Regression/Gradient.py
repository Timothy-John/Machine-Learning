import numpy as np

from sigmoid import sigmoid

#.....Gradient.....
def Gradient(X,y,theta):
    z=np.dot(X,theta)
    hx=sigmoid(z)
    gra=np.zeros([X.shape[1],1])
    m=len(y)
    gra=np.dot(X.T,hx-y)/m
    return gra