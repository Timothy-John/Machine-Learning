import scipy.optimize as scio
import numpy as np

from CostFunction import CostFunction

#.....Training with SCIPY with OneVsAll method.....
def OneVsAll(theta,X,y,lam,nums):
     n=X.shape[1]
     all_theta=np.zeros((nums,n))
     print("\nTRAIN NO.       COST:")
     for i in range(1,nums+1):       # nums+1 bcz y starts from 1 and ends in 10 
          theta = np.zeros((X.shape[1],1))
          res=scio.minimize(CostFunction,theta,method='CG',args=(X,y==i,lam),options={'maxiter':50}) # fmin_cg method
          all_theta[i-1]=res.x    #array starts with 0
          print("label ",i,"  :  ",res.fun)
     return all_theta
