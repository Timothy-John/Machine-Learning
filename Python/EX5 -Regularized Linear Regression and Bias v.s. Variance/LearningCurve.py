import numpy as np
import matplotlib.pyplot as plt

from optimize_scio import optimize_scio
from CostFunction import CostFunction

#.....Learning Curve.....
def LearningCurve(X,y,Xval,yval,lam):
    m=len(y)
    error_train = np.zeros([m,1])
    error_val = np.zeros([m,1])
    theta=np.zeros(X.shape[1])
    for i in range(1,m+1):
        theta,_=optimize_scio(theta,X[:i,:],y[:i],lam)
        error_train[i-1],_=CostFunction(theta,X[:i,:],y[:i],lam)  # Evaluate for i training sets
        error_val[i-1],_=CostFunction(theta,Xval,yval,lam)          # Evaluate for entire validation set...!! IMP
    
    print("\n\n# \tTrain Error\tValidation Error\n")
    for i in range(0,m):
        print('%d\t%f\t%f\n' % (i+1,error_train[i],error_val[i]))    
            
    plt.plot(range(m),error_train,label='Train Set')    
    plt.plot(range(m),error_val,label='Validation Set')             # can use np.linspace(0,m,m) instead of loop range
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()