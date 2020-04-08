import numpy as np 
import matplotlib.pyplot as plt
import random

#.....Visualizing Data.....
def DisplayData(X):
     m,n=np.shape(X)                     
     min,max=0,32                     #for jumping to next set of rows of image
     pos=random.randint(0,m)          #for random selection of data to display
     disp_array=np.zeros([320,320])   #initializing array for display [(32*10)X(32*10)]
     k=0                              #loop through the dataset to display a number
     for _ in range(0,10):            #loop number of rows to display
             for i in range(0,320):          #loop column (320 pixels.ie,10 images)
                 for j in range(min,max):    #loop rows (320 pixels//row of image)
                        disp_array[j][i]=X[pos][k]
                        k=k+1
                        if k>=n :                     #Re-initialization 
                            k=0
                            pos=random.randint(0,m)
                            break
             min=min+32
             max=max+32          
     plt.imshow(disp_array,cmap='gray')