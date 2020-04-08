import numpy as np

#.....Finds Closest Centroid to each Example of X.....
def ClosestCentroid(X,centroids):
    id_X=[]
    
    for i in range(len(X)):
        costs=np.sum(pow((X[i]-centroids),2),axis=1)  # if axis is not set, it will take default axis=None, which sums the whole dataset
        id_X.append(costs.argmin()+1)   # argmin() returns the position of array according to the axis which it is set
    return id_X
