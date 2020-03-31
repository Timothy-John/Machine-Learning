import numpy as np
import matplotlib.pyplot as plt

#.....Dispay a random number to predict.....
def DisplayData_nn(X):                 # X gets inverted when passing through function
    n=len(X)
    disp_array=np.zeros([20,20])
    k=0
    for i in range(0,20):       # 20 pixel row
        for j in range(0,20):   # 20 pixel column
            disp_array[j][i]=X[k]
            k=k+1
            if k>=n:
                break
    plt.imshow(disp_array,cmap='gray')     
    plt.show()    
