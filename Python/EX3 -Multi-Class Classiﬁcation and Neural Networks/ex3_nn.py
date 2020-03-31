import scipy.io as scio
import numpy as np
import random

from prediction_nn import prediction_nn
from DisplayData_nn import DisplayData_nn

#.....Neural Network Parameters.....
input_layer=401
hidden_layer=25
output_layer=10

#.....Load Data.....
data=scio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
m,n=X.shape
weights=scio.loadmat('ex3weights.mat')
theta1=weights['Theta1']
theta2=weights['Theta2']

#.....Prediction : Feed-Forward Neural Network Algorith.....
count=0
print("\nRunning Feed-Forward........")
print("Accuracy of Algorithm : ")
for i in range(5000):
     p=prediction_nn(X[[i],:],theta1,theta2)
     if p==y[i]:
         count+=1
print((count/5000)*100)         

#.....Implementing Feed-Forward NN.....
print("\nPredicting values in Dataset.....")
for i in range(0,m):
    pos=random.randint(0,m)
    DisplayData_nn(X[pos,:])
    pred=prediction_nn(X[[pos],:],theta1,theta2)
    print("Predicted Value: ",pred)
    ans=input("Press q to Quit.....")
    if ans=='q':
        break
