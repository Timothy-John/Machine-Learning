import numpy as np

#.....Normalization.....
def FeatureNormalize(X):
    mu=np.mean(X,axis=0)         # axis should be set to 0, because we store polynomial features in coulmn....IMP!!
    std=np.std(X,axis=0,ddof=1)  # axis=0 means consider column-wise and axis=1 means row-wise // mean and std should be done column wise... ddof is given for standard deviations to compute correctly
    X_norm=(X-mu)/std          
    return X_norm,mu,std
