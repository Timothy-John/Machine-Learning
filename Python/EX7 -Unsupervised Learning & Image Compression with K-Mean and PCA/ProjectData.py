import  numpy as np

#.....Projecting Data to dimension k.....
def ProjectData(X,U,k):
    U_redu=U[:,:k]
    z=np.dot(X,U_redu)
    return z