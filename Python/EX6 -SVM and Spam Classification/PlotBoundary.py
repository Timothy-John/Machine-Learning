import matplotlib.pyplot as plt
import numpy as np

from PlotData import PlotData

#.....Visializing Boundary.....
def PlotBoundary(model,X,y):

    if model.get_params()['kernel']=='linear':
        w=model.coef_[0]
        m=-w[0]/w[1]
        xx=np.linspace(min(X[:,0]),max(X[:,0]))    # can be given any integers instead of min and max... just to adjust range of fit line
        yy=m*xx-(model.intercept_[0]/w[1])
        PlotData(np.c_[X,y],'Positive','Negetive')
        plt.plot(xx,yy)
        plt.show()
    
    if model.get_params()['kernel']=='rbf':
        x1_plot=np.linspace(min(X[:,0]),max(X[:,0])+0.2)
        x2_plot=np.linspace(min(X[:,1])-0.2,max(X[:,1]))   # added and subtracted 0.2 to clearly see the Boundary
        x1,x2=np.meshgrid(x1_plot,x2_plot)

        vals=np.zeros(np.shape(x1))
        for i in range(x1.shape[1]):
            vals[:,i]=model.predict(np.c_[x1[:,i],x2[:,i]])
        PlotData(np.c_[X,y],'Positive','Negetive')    
        plt.contour(x1,x2,vals)
        plt.show()
"""
NOTE:  SKLEARN returns SLOPE-INTERCEPT form
       [a*x^n]+[b*x^(n-1)]+[c*x^(n-2)]+.... is the equation of line of fit
   ....since this is lenear coefficients of x are 'a' and 'b'  and the constant is 'c'
       training with sklearn returns a value which contains coefficients and intercepts 
       coef_ returns coefficients 'a' and 'b' [here]....therefore slope of fit line is: m=(-b/a)
       intercept_ returns constant 'c'....therefore intercept is (-c/a)  in y=mx+c
        m and c are being calculated in the above code
       not only for svm but also for sklearn.linear_model, this applies. (but not for Gaussian Kernel)
"""     
"""
In linear: we set a range of line to fit line
In Gaussian: we set a range of area to fit shape
"""