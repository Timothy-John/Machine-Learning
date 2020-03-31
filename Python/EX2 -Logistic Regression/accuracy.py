import numpy as np

from sigmoid import sigmoid

#.....Accuracy.....
def accuracy(X,theta):
    p=np.zeros(X.shape[0])
    p=sigmoid(np.dot(X,theta))>=0.5
    return p
