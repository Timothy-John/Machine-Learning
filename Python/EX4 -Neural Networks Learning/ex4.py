import scipy.io as scio
import numpy as np 
import matplotlib.pyplot as plt

from DisplayData import DisplayData
from nnCostFunction import nnCostFunction
from trainNN import trainNN
from rand_params import rand_params
from nnGradientDescent import nnGradientDescent
from predict import predict
from GradientCheck import GradientCheck

#.....Neural Network Parameters.....
input_layer=400
hidden_layer=25
output_layer=10

#.....Load Data.....
data=scio.loadmat('ex4data1.mat')
X=data['X']
y=data['y']
m,n=X.shape

weights=scio.loadmat('ex4weights.mat')
theta1=weights['Theta1']     # (25, 401)
theta2=weights['Theta2']     # (10, 26)

nn_params=np.hstack((theta1.reshape(-1),theta2.reshape(-1)))  # UnRolling parameters to a 1-D array to optimize with scipy
#nn_params = np.r_[theta1.ravel(), theta2.ravel()] ....same as above   # (10285,)     
lam=0

#.....Display Data.....
print("Visualizing Data......")
DisplayData(X)

#.....Cost Functon..... 
J,_=nnCostFunction(nn_params,input_layer,hidden_layer,output_layer,X,y,lam)  # Cost & Gradient returned
print("\nCost (Without Regularizaton) of given Thetas: ",J)
lam=1
J,_=nnCostFunction(nn_params,input_layer,hidden_layer,output_layer,X,y,lam)  # Cost & Gradient returned
print("Cost (With Regularizaton) of given Thetas   : ",J)

#.....Random Thetas.....
rand_nn_params=rand_params(input_layer,hidden_layer,output_layer)

#.....Training with Gradient Descent.....  
print("\n   Training Neural Network with Gradient Descent.......")
iterations=200
alpha=0.01
all_theta,J_history=nnGradientDescent(X,y,rand_nn_params,iterations,alpha,input_layer,hidden_layer,output_layer,lam)
J,_=nnCostFunction(all_theta,input_layer,hidden_layer,output_layer,X,y,lam)
print("> Cost of trained Thetas : ",J)

its=np.zeros([iterations,1])
for i in range (iterations):
    its[i]=i
plt.plot(its,J_history)    # Plotting (Cost vs Iterations) Convergence Graph
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

#.....Gradient Check.....
# epsilion method to check if Back-Propogation is implemented correctly
GradientCheck(nn_params,input_layer,hidden_layer,output_layer,X,y,lam)

#.....Training Neural Network with SCIPY.....
rand_nn_params=rand_params(input_layer,hidden_layer,output_layer)    # assigning random thetas
print("\n   Training Neural Network with SCIPY........")
all_theta=trainNN(rand_nn_params,X,y,lam,input_layer,hidden_layer,output_layer)

#.....Accuracy.....
count=0
for i in range(m):
    p=predict(nn_params,X[[i],:],input_layer,hidden_layer,output_layer)     # nn_params is pre-trained
    if p==y[i]:
        count+=1
print("\nAccuracy with pre-trained Thetas                : ",(count/m)*100)

count=0
for i in range(m):
    p=predict(all_theta,X[[i],:],input_layer,hidden_layer,output_layer)     # all_theta is newly trained
    if p==y[i]:
        count+=1
print("Accuracy with newly-trained Thetas (with SCIPY) : ",(count/m)*100)
  
