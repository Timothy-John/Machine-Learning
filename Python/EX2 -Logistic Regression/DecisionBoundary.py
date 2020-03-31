import numpy as np
import matplotlib.pyplot as plt

from PlotData import PlotData
from sigmoid import sigmoid
from MapFeature import MapFeature

#.....Decision Boundary.....
def DecisionBoundary(data,theta):
     PlotData(data,'Exam Score 1','Exam Score 2','Admitted','Not Admitted')
     X=data[:,0:2]
     x1_min, x1_max = X[:,0].min(), X[:,0].max()
     x2_min, x2_max = X[:,1].min(), X[:,1].max()
     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
     h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta))
     h = h.reshape(xx1.shape)
     plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

#.....Decision Boundary for ex2_reg.....
def DecisionBoundary_reg(data,theta):
     PlotData(data,'Microchip Test 1','Microchip Test 2','Accepted','Not Accepted')
     X=data[:,0:2]
     x1_min, x1_max = X[:,0].min(), X[:,0].max()
     x2_min, x2_max = X[:,1].min(), X[:,1].max()
     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
     XX=MapFeature(np.c_[xx1.ravel(), xx2.ravel()])
     XXX=np.hstack((np.ones((XX.shape[0],1)),XX))        
     h = sigmoid(XXX.dot(theta))   #only difference from DecisionBoundary() is addition of MapFeature.....
     h = h.reshape(xx1.shape)      #.....same as MapFeature(X)*Theta
     plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
