import numpy as np

#.....Sigmoid Function.....
def sigmoid(z) :
    h=1/(1+np.exp(-z))
    return h
