import numpy as np

from Multivariate_Gaussian import MultivariateGaussian
from GaussianParams import GaussianParams

#.....Selecting BEst Epsilon.....
def Best_Epsilon(Xval,yval):
    best_epsilon=0
    best_F1=0
    F1=0
    
    mu,std=GaussianParams(Xval)
    Pval=MultivariateGaussian(Xval,mu,std)
    Pval=Pval[:,np.newaxis]
    steps=(max(Pval)-min(Pval))/1000   # jumps to be taken for loop

    for epsilon in np.arange(min(Pval),max(Pval),steps):    # arange has to be used instead of range
        TP = np.sum( Pval[yval==1] < epsilon)
        TN = np.sum( Pval[yval==0] >= epsilon)
        FP = np.sum( Pval[yval==0] < epsilon)
        FN = np.sum( Pval[yval==1] >= epsilon)

        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = (2*Precision*Recall)/(Precision+Recall)
        
        if F1>best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon,best_F1