import numpy as np

from sigmoid import sigmoid

#.......Feed Forward Algorithm.....
def prediction_nn(X,theta1,theta2):
    m=X.shape[0]
    p=np.zeros((m,1))
    a1=np.hstack((np.ones([m,1]),X))
    z2=np.dot(a1,theta1.T)
    a2=sigmoid(z2)
    a2=np.hstack((np.ones([m,1]),a2))
    z3=np.dot(a2,theta2.T)
    a3=sigmoid(z3)
    p=a3.argmax()+1
    return p
