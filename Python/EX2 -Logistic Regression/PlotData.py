import numpy as np
import matplotlib.pyplot as plt

#.....Plotting Data.....
def PlotData(data,xlabel,ylabel,legend1,legend2):
      X=data[:,0:2]
      y=data[:,2]
      """
      m=len(y)
      for i in range(m):
          if y[i]==1:
             plt.scatter(X[i,0],X[i,1],color='green',marker='+',label='pos')
          else:
             plt.scatter(X[i,0],X[i,1],color='red',label='neg')         
      """
      pos= y==1
      neg= y==0
      plt.scatter(X[pos,0],X[pos,1],color='green',marker='+',label=legend1)
      plt.scatter(X[neg,0],X[neg,1],color='red',label=legend2)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.legend(loc='upper right')