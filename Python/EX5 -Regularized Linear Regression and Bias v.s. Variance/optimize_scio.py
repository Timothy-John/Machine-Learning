import scipy.optimize as scio

from CostFunction import CostFunction
#.....Optimization with SCIPY.....
def optimize_scio(theta,X,y,lam):
    res=scio.minimize(CostFunction,theta,method='CG',args=(X,y,lam),jac=True) 
    # jac should be true if the cost computation has gradient returned
    theta=res.x           
    J=res.fun
    return theta,J
