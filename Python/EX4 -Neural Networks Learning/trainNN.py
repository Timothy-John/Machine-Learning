import scipy.optimize as scio
import numpy as np

from nnCostFunction import nnCostFunction

#.....Training NN with SCIPY.....
def trainNN(nn_params,X,y,lam,input,hidden,output):
    # y==i is calculated in nnCostFunction...NOT here, like in ex3.py
    res = scio.minimize(nnCostFunction,nn_params,method="CG",args=(input,hidden,output,X,y,lam),jac=True,options={'maxiter':50})

    all_theta = res.x
    print(">Cost of trained Thetas : ",res.fun)
    return all_theta
