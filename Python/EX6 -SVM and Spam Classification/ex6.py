import numpy as np
import scipy.io as scio
import sklearn.svm as skvm
import matplotlib.pyplot as plt

from PlotData import PlotData
from PlotBoundary import PlotBoundary
from BestParams import BestParams

#.....Loading Dataset 1.....
data1=scio.loadmat('ex6data1.mat')
X=data1['X']
y=data1['y']
y=y.ravel()        # Only to avoid Warnings

#.....Plotting Data.....
      # converting into a matrix by (concatinating/stacking) X and y with np.c_ or np.hstack
PlotData(np.c_[X,y],'Positive','Negetive')   
plt.show()

#.....Training with SVM using SKLEARN library.....
print("\nTraining with SVM using LENEAR KERNEL.........")
svm=skvm.SVC(C=1,kernel='linear')      # SVC for classification;  SVR for regression
model=svm.fit(X,y)
PlotBoundary(model,X,y)

#.....Loading Dataset 2.....
data2=scio.loadmat('ex6data2.mat')
X=data2['X']
y=data2['y']
y=y.ravel()        # Only to avoid Warnings

#.....Plotting and Training with Gaussian Kernel.....
PlotData(np.c_[X,y],'Positive','Negetive') 
plt.show()
print("Training with SVM using GAUSSIAN KERNEL.......")
svm=skvm.SVC(C=1,kernel='rbf',gamma=100)    # gamma decides the complexity of shape // C decides how well to fit the data (variance)
model=svm.fit(X,y)                          # gamma is just the inverse of sigma
PlotBoundary(model,X,y)

#.....Loading Dataset 3.....
data3=scio.loadmat('ex6data3.mat')
X=data3['X']
y=data3['y']
y=y.ravel()        # Only to avoid Warnings
Xval=data3['Xval']
yval=data3['yval']

#.....Plotting and training with Gaussian kernel.....
PlotData(np.c_[X,y],'Positive','Negetive')
plt.show()

#.....Function to select best C and gamma.....
C,gamma=BestParams(X,y,Xval,yval)

#.....Training with Gaussian Kernel.....
print("Training with SVM using GAUSSIAN KERNEL.......")
svm=skvm.SVC(C=C,kernel='rbf',gamma=gamma)
model=svm.fit(X,y)
PlotBoundary(model,X,y)