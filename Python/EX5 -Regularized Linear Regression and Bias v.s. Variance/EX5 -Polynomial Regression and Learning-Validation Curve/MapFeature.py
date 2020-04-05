import numpy as np

#.....Map Featuring.....
def MapFeature(X,p): 
    X=X[:,np.newaxis]
    X1=np.zeros([X.shape[0],p])
    for i in range(0,p):
        X1[:,[i]]=pow(X,i+1)
    return X1
