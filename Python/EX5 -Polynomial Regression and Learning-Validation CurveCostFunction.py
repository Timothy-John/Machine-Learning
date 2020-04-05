import numpy as np

#.....Cost Function.....
def CostFunction(theta,X,y,lam):
    m=len(y)
    theta=theta[:,np.newaxis]
    theta_zero = np.concatenate(([0],theta[1:,0]))  # intercept term should be 0 for regularization
    theta_zero=theta_zero[:,np.newaxis]

    h=np.dot(X,theta)
    J=(np.sum(np.square(h-y))+lam*np.sum(np.square(theta_zero)))/(2*m)
    grad=(np.dot(X.T,(h-y))+lam*theta_zero)/m	
    return J,grad.reshape(-1)         # reshaping is important for scipy optimizer to work 
