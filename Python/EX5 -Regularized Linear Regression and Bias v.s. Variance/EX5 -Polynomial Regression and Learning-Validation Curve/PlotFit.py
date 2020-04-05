import numpy as np

from MapFeature import MapFeature

#.....Plot Fit for Polynomial Features.....
def PlotFit(data,p,mu,std):
    x = np.arange(min(data['X']) - 15, max(data['X']) + 25, 0.05)  # defining a range of array for plot-fit line
    x_poly=MapFeature(x,p)
    x_poly=(x_poly-mu)/std        # should be Normalized with mean and std of X_poly in ex5.py (not x_poly)
    x_poly=np.hstack((np.ones([len(x_poly),1]),x_poly))          # intercept term should be added after Map-Featuring
    return x,x_poly