import numpy as np

#.....Gaussian Parameters mu and co_var.....
def GaussianParams(X):
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    return mu,std