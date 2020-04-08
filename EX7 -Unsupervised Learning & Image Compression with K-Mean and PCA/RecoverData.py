import numpy as np

#.....Data Recovery.....
def RecoverData(Z,U,k):
    U_redu=U[:,:k]
    X_recov=np.dot(Z,U_redu.T)
    return X_recov