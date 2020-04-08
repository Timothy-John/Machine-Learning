import numpy as np

#.....Random Initialization of Centroids.....
def Random_Init(X,k):
    return X[np.random.choice(X.shape[0], k)]