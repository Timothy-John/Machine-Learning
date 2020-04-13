import numpy as np

#.....Normalizing Ratings.....
def Normalize(Y,R):
    Y_mean= np.zeros([Y.shape[0],1])
    Y_norm= np.zeros(Y.shape)
    for i in range(len(Y)):
        id= np.where(R[i,:]==1)
        Y_mean[i,:]= np.mean(Y[i,id],axis=1)
        Y_norm[i,id]= Y[i,id] -Y_mean[i,:]
    return Y_norm,Y_mean