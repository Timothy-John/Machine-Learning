import numpy as np

#.....Feature Normalizing.....
def FeatureNormalize(X):
    X=(X-np.mean(X))/(np.std(X))
    return X