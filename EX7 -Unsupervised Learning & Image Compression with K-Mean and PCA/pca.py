import numpy as np

#.....Principle Component Analysis.....
#PCA works in two steps: 
# 1) Compute Covariance Matrix (sigma)  
# 2) Find Eigen Matrix and values of Covariance Matrix
def pca(X):
    m=len(X)
    sigma=(np.dot(X.T,X))/m
    U,S,V=np.linalg.svd(sigma)
    return U,S