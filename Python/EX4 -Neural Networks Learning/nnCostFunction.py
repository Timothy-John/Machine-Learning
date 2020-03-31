import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

#.....NN Cost Function.....
def nnCostFunction(nn_params,input,hidden,output,X,y,lam):   
     m=X.shape[0]
     # Rolling parameters to original shape
     theta1=nn_params[0:hidden*(input+1)].reshape(hidden,input+1)       # (25, 401)
     theta2=nn_params[(hidden*(input+1)):].reshape(output,hidden+1)     # (10, 26)

     Y=np.zeros([m,10])  # here, y is nums from 1-10...but, sigmoid(y) <1 & >0 ...
     for i in range(m):  #......i will be value from 1-10 from y.....  
          Y[i,y[i]-1]=1  #.....use either loop to run over all examples or use Y matrix 5000X10 each row of labels 1-10

     D1 = np.zeros_like(theta1)
     D2 = np.zeros_like(theta2)

     a1= np.hstack((np.ones([m,1]),X))             # 5000 x 401                                      
     z2=a1.dot(theta1.T)                           # 5000 x 25 
     a2=np.hstack((np.ones([m,1]),sigmoid(z2)))    # 5000 x 26                
     z3=a2.dot(theta2.T)                           # 5000 x 10
     a3=sigmoid(z3)
     hx=a3               # Better Intution

     # Calculate Gradient
     d3 =  hx - Y        # 5000 x 10
     d2 = (np.dot(theta2.T,d3.T)[1:]).T*(sigmoidGradient(z2))  # 5000 x 25
     # Accumulate errors for gradient calculation
     D1 = D1 + np.dot(d2.T,a1) # 25 x 401 (matches theta0)  D1+... is imp!!
     D2 = D2 + (np.dot(d3.T,a2)) # 10 x 26 (matches theta1) D2+... is imp!!

     J=np.sum(Y*np.log(hx)+(1-Y)*np.log(1-hx))/(-m)              # np.dot should not be used...use *
     J=J+(lam*(np.sum(pow(theta1,2))+np.sum(pow(theta2,2))))/(2*m)   # Regularization... np.square instead of pow()

     grad1 = (1.0/m)*D1 + (lam/m)*theta1           # (25, 401)
     grad1[0] = grad1[0] - (lam/m)*theta1[0]
     
     grad2 = (1.0/m)*D2 + (lam/m)*theta2           # (10, 26)
     grad2[0] = grad2[0] - (lam/m)*theta2[0]

     grad = np.append(grad1,grad2).reshape(-1)  # scipy.optimize will not work without gradient
     return J,grad

"""
#.....NN Cost Function.....
def nnCostFunction(nn_params,input,hidden,output,X,y,lam):   
     m=X.shape[0]

     # Rolling parameters to original shape
     theta1=nn_params[0:hidden*(input+1)].reshape(hidden,inp+1)         # (25, 401)
     theta2=nn_params[(hidden*(input+1)):].reshape(output,hidden+1)     # (10, 26)

     Y=np.zeros([m,10])  # here, y is nums from 1-10...but, sigmoid(y) <1 & >0 ...
     for i in range(m):  #......i will be value from 1-10 from y.....  
          Y[i,y[i]-1]=1  #.....use either loop to run over all examples or use Y matrix 5000X10 each row of labels 1-10

          # Add column of ones to X
     ones = np.ones((m,1)) 
     d = np.hstack((ones,X))# add column of ones

     cost = [0]*m
     # Initalize gradient vector
     D1 = np.zeros_like(theta1)
     D2 = np.zeros_like(theta2)
     for i in range(m):
          
          a1 = d[i][:,None] # 401 x 1
          z2 = np.dot(theta1,a1) # 25 x 1 
          a2 = sigmoid(z2) # 25 x 1
          a2 = np.vstack((np.ones(1),a2)) # 26 x 1
          z3 = np.dot(theta2,a2) #10 x 1
          h = sigmoid(z3) # 10 x 1
          a3 = h # 10 x 1
          cost[i] = (np.sum((-Y[i][:,None])*(np.log(h)) - (1-Y[i][:,None])*(np.log(1-h))))/m
          # Calculate Gradient
          d3 = a3 - Y[i][:,None]
          d2 = np.dot(theta2.T,d3)[1:]*(sigmoidGradient(z2))
          # Accumulate errors for gradient calculation
          D1 = D1 + np.dot(d2,a1.T) # 25 x 401 (matches theta0)
          D2 = D2 + np.dot(d3,a2.T) # 10 x 26 (matches theta1)

     reg = (lam/(2*m))*((np.sum(theta1[:,1:]**2)) + (np.sum(theta2[:,1:]**2)))

     grad1 = (1.0/m)*D1 + (lam/m)*theta1
     grad1[0] = grad1[0] - (lam/m)*theta1[0]
     
     grad2 = (1.0/m)*D2 + (lam/m)*theta2
     grad2[0] = grad2[0] - (lam/m)*theta2[0]
     grad = np.append(grad1,grad2).reshape(-1)
     J=cost+reg
     return sum(J),grad
"""
