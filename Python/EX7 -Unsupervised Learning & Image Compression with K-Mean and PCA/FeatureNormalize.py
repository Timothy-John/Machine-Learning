import numpy as np

#.....Feature Normalizing.....
def FeatureNormalize(X):
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X=(X-mu)/std
    return X