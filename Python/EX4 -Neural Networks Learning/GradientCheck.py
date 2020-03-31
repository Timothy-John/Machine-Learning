import numpy as np

from nnCostFunction import nnCostFunction

#.....Gradient Check.....
def GradientCheck(nn_params,input,hidden,output,X,y,lam):
    epsilion=pow(10,-4)
    _,grad=nnCostFunction(nn_params,input,hidden,output,X,y,lam)
    param_plus=nn_params+epsilion
    param_minus=nn_params-epsilion
    J1,_=nnCostFunction(param_plus,input,hidden,output,X,y,lam)
    J2,_=nnCostFunction(param_minus,input,hidden,output,X,y,lam)
    check=(J1-J2)/(2*epsilion)
    print("\nChecking Gradients.....")
    print(np.sum(grad),"   ~   ",check)
    