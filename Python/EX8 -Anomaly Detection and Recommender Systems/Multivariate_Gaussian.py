import numpy as np

#.....Multivariate Gaussian.....
def MultivariateGaussian(X,mu,co_var):
    n=mu.size
    if co_var.ndim==1 or co_var.shape[0] == 1 or co_var.shape[1]==1:
        co_var=np.diag(co_var)

    P=(np.exp(-np.sum(np.square(np.dot((X-mu),np.linalg.pinv(co_var))),axis=1)/2))/(pow(2*np.pi,n*0.5)*pow(np.linalg.det(co_var),0.5))
    return P