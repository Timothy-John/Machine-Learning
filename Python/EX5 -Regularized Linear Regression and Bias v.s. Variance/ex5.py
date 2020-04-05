import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

from PlotData import PlotData
from CostFunction import CostFunction
from optimize_scio import optimize_scio
from LearningCurve import LearningCurve
from MapFeature import MapFeature
from FeatureNormalize import FeatureNormalize
from PlotFit import PlotFit
from CrossValidation import CrossValidation

#.....Load Data.....
data=scio.loadmat('ex5data1.mat')
X=data['X']
y=data['y']
Xval=data['Xval']
yval=data['yval']
Xtest=data['Xtest']
ytest=data['ytest']

theta=np.ones(2)
lam=0      # Change Me!
alpha=0.1
iterations=1500
m=len(X)
m2=len(Xval)
m3=len(Xtest)

X=np.hstack((np.ones([m,1]),X))
Xval=np.hstack((np.ones([m2,1]),Xval))
Xtest=np.hstack((np.ones([m3,1]),Xtest))

#.....Plotting Data.....
PlotData(data['X'],y)
plt.show()

#.....Cost Function && Gradient.....
h=np.dot(X,theta)
J,grad=CostFunction(theta,X,y,lam)
print("Cost : ",J)
print("Gradient : \n",grad)

#.....Optimization with SCIPY.....
print("\nTraining........")
theta,J=optimize_scio(theta,X,y,lam)         
print("\nTheta : ",theta)
print("Cost with Theta : ",J)

PlotData(data['X'],y)
plt.plot(data['X'],np.dot(X,theta),'g')
plt.show()

#.....Plotting Learning Curve.....
LearningCurve(X,y,Xval,yval,lam)

#.....Polynomial Featuring.....
print("\nUsing Polynomial Features.......")
p=5    # Change Me!
                           # TRAIN SET
X_poly=MapFeature(X[:,1],p)
X_poly,mu,std=FeatureNormalize(X_poly)
X_poly=np.hstack((np.ones([m,1]),X_poly))          # add intercept term after map-featuring
                           # VALIDATION SET
Xval_poly=MapFeature(Xval[:,1],p)
Xval_poly=(Xval_poly-mu)/std
Xval_poly=np.hstack((np.ones([m2,1]),Xval_poly))
                           # TEST SET
Xtest_poly=MapFeature(Xtest[:,1],p)
Xtest_poly=(Xtest_poly-mu)/std
Xtest_poly=np.hstack((np.ones([m3,1]),Xtest_poly))

#.....Cost Function.....
lam=0    # Change Me!
theta=np.zeros(X_poly.shape[1])
J,_=CostFunction(theta,X_poly,y,lam)
print("Cost     : ",J)

#.....Optimizing Theta.....
print("\nTraining........")
theta,J=optimize_scio(theta,X_poly,y,lam)
print("Cost with Trained Theta : ",J)

x1,x1_poly=PlotFit(data,p,mu,std)
PlotData(data['X'],y)
plt.plot(x1,np.dot(x1_poly,theta),'g-')
plt.show()

#.....Learning Curve(Polynimial Features).....
LearningCurve(X_poly,y,Xval_poly,yval,lam)

#.....Validation Curve.....
lambda_vectors=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
CrossValidation(X_poly,y,Xval_poly,yval,lambda_vectors)