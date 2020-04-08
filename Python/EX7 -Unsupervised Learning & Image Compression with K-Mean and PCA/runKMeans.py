import numpy as np
import matplotlib.pyplot as plt

from ClosestCentroid import ClosestCentroid
from ComputeCentroid import ComputeCentroid

#.....K-Mean Algorithm.....
    #K-Mean run in two steps- Cluster Assignment and Moving Centroid
def runKMeans(X,centroids,max_iters,show_plot):
    k=centroids.shape[0]
    previous_centroids=centroids

    for _ in range(max_iters):
        id_X = ClosestCentroid(X,centroids)                               # CLUSTER ASSIGNMENT

        if show_plot==1:
            plt.plot(X[:,0],X[:,1],'go')
            plt.plot(centroids[:,0],centroids[:,1],'rx')
            plt.plot(previous_centroids[:,0],previous_centroids[:,1],'bx')
            plt.show()

        previous_centroids=centroids
        centroids = ComputeCentroid(X,id_X,k)                             # MOVING CENTROIDS
    return id_X,centroids