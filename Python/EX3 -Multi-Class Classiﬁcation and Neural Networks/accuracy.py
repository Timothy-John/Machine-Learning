import numpy as np

from sigmoid import sigmoid

#.....Accuracy.....
def accuracy(theta,X):
    p=np.zeros([5000,1])
    predict=sigmoid(np.dot(X,theta.T))
    p=np.argmax(predict)+1
    return p
    
