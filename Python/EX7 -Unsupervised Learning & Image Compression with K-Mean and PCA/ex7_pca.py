import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

from FeatureNormalize import FeatureNormalize
from pca import pca
from ProjectData import ProjectData
from RecoverData import RecoverData
from DisplayData import DisplayData

#.....Loading Dataset.....
data1=scio.loadmat('ex7data1.mat')
X_raw=data1['X']

#.....Plotting Data.....
print("\nPlotting Dataset.......")
plt.scatter(X_raw[:,0],X_raw[:,1])
plt.show()

X=FeatureNormalize(X_raw)        #.....Normalizing Dataset.....

#.....Principle Component Analysis Implementation.....
U,S = pca(X)
k=1     # dimension to which project to
Z=ProjectData(X,U,k)             # Z is Projected X
X_recovered=RecoverData(Z,U,k)

plt.scatter(X[:,0],X[:,1])
plt.plot(X_recovered[:,0],X_recovered[:,1],'r*')
plt.show()

#.....Running PCA on Face Datas.....
data2=scio.loadmat('ex7faces.mat')
X_faces=data2['X']
DisplayData(X_faces)
plt.title('Original Data')
plt.show()

X=FeatureNormalize(X_faces)
U,S = pca(X)
k=100
Z=ProjectData(X,U,k)
X_recovered = RecoverData(Z,U,k)

DisplayData(X_recovered)
plt.title('Data after Dimension Reduction')
plt.show()