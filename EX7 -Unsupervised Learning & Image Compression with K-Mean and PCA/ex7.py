import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from ClosestCentroid import ClosestCentroid
from ComputeCentroid import ComputeCentroid
from runKMeans import runKMeans
from Random_Init import Random_Init

#.....Loading Data.....
data=scio.loadmat('ex7data2.mat')
X=data['X']
k=3  # No:of centroids.....CHANGE ME!

#.....Checking ClosestCentroid().....
example_centroids=np.array([[3,3],[6,2],[8,5]])
id_X=ClosestCentroid(X,example_centroids)
print("\nCentroids of First 3 examples : ",id_X[:3])

#.....Checking ComputeCentroid().....
new_centroids=ComputeCentroid(X,id_X,k)
print("New Centroids : \n",new_centroids)

#.....Plotting Dataset.....
print("\nVisializing Data.......")
plt.plot(X[:,0],X[:,1],'go')
plt.show()

#....K-Mean.....
print("\nRunnng K-means.......")
initial_centroids=np.array([[3,3],[6,2],[8,5]])
max_iters=5
id_X,centroids=runKMeans(X,initial_centroids,max_iters,show_plot=1)

#.....Image Compression with K-Means.....
print("\nRunning K-Mean on Pixels.........")
A = plt.imread('bird_small.png')            # you can also load bird_small.mat into A
original_shape = np.shape(A)
plt.imshow(A)
plt.title('Original Image')
plt.show()

#.....Random Initialization.....
X = A.reshape(A.shape[0]*A.shape[1],3)      # reshape A to get RGB values of each pixels     [(128*128) x 3]
k=16             # No: of Centroids.....CHANGE ME!
max_iters=10
initial_centroids = Random_Init(X,k)

#.....Compressing Pixels with K-Mean.....
id_X,centroids = runKMeans(X,initial_centroids,max_iters,show_plot=0)
id_X=[x-1 for x in id_X]                    # 1 is subtracted from every element of id_X because it starts from 1 and will not take element at position 0
A_reduced = centroids[id_X,:]
A_reduced = A_reduced.reshape(original_shape)


f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.imshow(A)
ax2.imshow(A_reduced)
plt.title('Compressed Image')
plt.show()

#NOTE: Elbow Method can be used to Select Optimum no: of Centroids 