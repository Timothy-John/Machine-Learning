import numpy as np

#.....Finds New Centroids with X assigned to each existing centroids by ClosestCentroid.py().....
def ComputeCentroid(X,id_X,k):
    new_centroids = np.zeros((k,np.size(X,1)))
    aug_X = np.hstack((np.array(id_X)[:,None],X))
    for i in range(1,k+1):
        new_centroids[i-1] = np.mean(X[aug_X[:,0] == i], axis=0)
    
    return new_centroids


# Same as above but more premitive code
"""
new_centroids = np.zeros((k,np.size(X,1)))
for i in range(1,k+1):
    sum=np.zeros([1,2])
    count=0
    for j in range(len(X)):
        if id_X[j]==i:
            count+=1
            sum=sum+X[j]
    new_centroids[i-1]=(sum/count)
return new_centroids
"""