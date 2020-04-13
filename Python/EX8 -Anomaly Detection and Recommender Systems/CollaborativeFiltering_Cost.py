import numpy as np

#.....Cost Function for Collabrative Filtering.....
def Collab_Filt_Cost(params,Y,R,num_users,num_movies,num_features,lam):
    X= params[:num_movies *num_features].reshape(num_movies,num_features)
    theta= params[num_movies *num_features:].reshape(num_users,num_features)

    J= np.sum(np.square(np.dot(X,theta.T)-Y) *R)/2
    reg= lam/2 * (np.sum(np.square(X)) + np.sum(np.square(theta)))
    
    theta_grad= np.dot(((np.dot(X,theta.T)-Y) *R).T,X) + (lam*theta)
    X_grad= np.dot((np.dot(X,theta.T)-Y) *R,theta) + (lam*X)  
    grad= np.hstack((X_grad.flatten(),theta_grad.flatten()))  
    
    return (J + reg), grad.flatten()