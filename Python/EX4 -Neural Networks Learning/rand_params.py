import numpy as np
import numpy.random as nr

#.....Generate Random Parameters.....
# random initilization with zero array gives high cost 
def rand_params(input,hidden,output):
    e=np.sqrt(6)/(np.sqrt(output+input))      # can be initialized with random,--> accuracy will be low
    theta1_shape = (hidden, input+1)
    theta2_shape = (output, hidden+1)
    rand_nn_params = np.hstack(([((2*e*nr.random(theta1_shape))-e).reshape(-1),((2*e*nr.random(theta2_shape))-e).reshape(-1)]))  # UnRolling
    return rand_nn_params
