import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from GaussianParams import GaussianParams
from Multivariate_Gaussian import MultivariateGaussian
from GaussianFit import GaussianFit
from Best_Epsilon import Best_Epsilon

#NOTE: Gaussian Distribution can be used as Unsupervised Algorithm without use of 'y'
#....however, if you want to compute best epsilon computationally (using for loop), you would need y matrix
#....you can select epsilon manually with plots also. In that case you don't need a y matrix and this algorith remains as an unsupervised learning algorithm

#.....Loading and Visializing Dataset.....
data1=scio.loadmat('ex8data1.mat')
X=data1['X']
Xval=data1['Xval']
yval=data1['yval']

plt.scatter(X[:,0],X[:,1])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

#.....Multivariate Gaussian Distribution.....
print("\nVisualizing Gaussian Fit.......")
mu,co_var=GaussianParams(X)
P = MultivariateGaussian(X,mu,co_var)

#.....Visualizing Gaussian Fit.....
GaussianFit(X,mu,co_var)
plt.show()

#.....Finding  Outliers.....
epsilon,F1=Best_Epsilon(Xval,yval)     # Validation set is usd for selecting best epsilon
print("\n# Validation Set")
print(" Best Epsilon Calculated : ",epsilon,"\n \t   with F1 score : ",F1)
print("   No: of Outliers found : ",np.sum(P < epsilon))

outliers=np.where(P < epsilon)
GaussianFit(X,mu,co_var)
plt.plot(X[outliers,0],X[outliers,1],'ro')
plt.show()

#.....MultiDimensional Outliers.....
data2=scio.loadmat('ex8data2.mat')
X_2=data2['X']
Xval_2=data2['Xval']
yval_2=data2['yval']

mu_2,co_var_2 = GaussianParams(X_2)
P_2 = MultivariateGaussian(X_2,mu_2,co_var_2)
epsilon_2,F1_2 = Best_Epsilon(Xval_2,yval_2)

print("\n# Validation Set : Multi-Dimensional Dataset")
print(" Best Epsilon Calculated : ",epsilon_2,"\n \t   with F1 score : ",F1_2)
print("   No: of Outliers found : ",np.sum(P_2 < epsilon))