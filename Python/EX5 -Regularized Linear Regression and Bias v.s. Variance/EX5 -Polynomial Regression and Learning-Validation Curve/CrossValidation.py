import numpy as np
import matplotlib.pyplot as plt

from optimize_scio import optimize_scio
from CostFunction import CostFunction

#.....Cross-Validation Curve.....
def CrossValidation(X,y,Xval,yval,lam_vec):
    error_train=[]
    error_val=[]
    for lam in lam_vec:
        initial_theta=np.zeros(X.shape[1])
        theta,_=optimize_scio(initial_theta,X,y,lam)
        error_train.append((CostFunction(theta,X,y,lam))[0])
        error_val.append((CostFunction(theta,Xval,yval,lam))[0])
    plt.plot(lam_vec,error_train,label='Train')
    plt.plot(lam_vec,error_val,label='Cross-Validation')
    plt.xlabel('Lamda Vectors')
    plt.ylabel('Cost')
    plt.title('Cross-Validation Curve')
    plt.legend()
    plt.show()