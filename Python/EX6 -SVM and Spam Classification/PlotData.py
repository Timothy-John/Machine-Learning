import matplotlib.pyplot as plt

#.....Plotting.....
def PlotData(data,legend1,legend2):
      X=data[:,0:2]
      y=data[:,2]
      pos= y==1
      neg= y==0
      plt.scatter(X[pos,0],X[pos,1],color='green',marker='+',label=legend1)
      plt.scatter(X[neg,0],X[neg,1],color='red',label=legend2)
      plt.legend()  