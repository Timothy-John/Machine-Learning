import matplotlib.pyplot as plt
import numpy as np

from Multivariate_Gaussian import MultivariateGaussian

#.....Gaussian Fit Line.....
# For any plotting: 
# 1) define the range of view
#       ...for 2-d plot define a range of 1-d array(arange()) to slide along the 2d area
#       ...for 3-d plot define a meshgrid (range of 2-d array) to slide along the 3d volume
# 2) then plot the values on the defined range view by passing the defined view to the function(here MultiGaussian()) and plotting....

def GaussianFit(X,mu,co_var):
    x,y=np.meshgrid(np.arange(5,25),np.arange(0,25))
    XX=np.vstack((x.flatten(),y.flatten()))    # reshape(-1) can be used instead of flatten()
    Z=MultivariateGaussian(XX.T,mu,co_var)
    Z=np.reshape(Z,x.shape)
    plt.contour(x,y,Z)
    plt.scatter(X[:,0],X[:,1])
    plt.title('Gaussian Fit Contour')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')