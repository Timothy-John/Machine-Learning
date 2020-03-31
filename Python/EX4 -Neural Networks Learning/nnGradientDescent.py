import numpy as np

from nnCostFunction import nnCostFunction

#.....Gradient Descent.....
def nnGradientDescent(X,y,nn_params,iterations,alpha,input,hidden,output,lam):
    theta1=nn_params[0:hidden*(input+1)].reshape(hidden,input+1)       # (25, 401)
    theta2=nn_params[(hidden*(input+1)):].reshape(output,hidden+1)     # (10, 26)
    J_history=np.zeros([iterations,1])
    for i in range (iterations):
        J_history[i],grad=nnCostFunction(nn_params,input,hidden,output,X,y,lam)
        grad1=grad[0:hidden*(input+1)].reshape(hidden,input+1)       # (25, 401)
        grad2=grad[(hidden*(input+1)):].reshape(output,hidden+1)     # (10, 26)
        theta1=theta1-alpha*grad1
        theta2=theta2-alpha*grad2
        nn_params=np.hstack((theta1.reshape(-1),theta2.reshape(-1)))
    return nn_params,J_history