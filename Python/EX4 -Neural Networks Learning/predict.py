import numpy as np

from sigmoid import sigmoid

#.....Prediction.....
def predict(nn_params,X,input,hidden,output):
    m=X.shape[0]
    theta1=nn_params[0:hidden*(input+1)].reshape(hidden,input+1)       # (25 x 401)
    theta2=nn_params[(hidden*(input+1)):].reshape(output,hidden+1)     # (10, 26)
     
    a1=np.hstack((np.ones([m,1]),X))
    z2=np.dot(a1,theta1.T)
    a2=np.hstack((np.ones([m,1]),sigmoid(z2)))
    z3=np.dot(a2,theta2.T)
    a3=sigmoid(z3)
    p=np.argmax(a3)+1
    
    return p