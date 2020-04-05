import numpy as np
import sklearn.svm as skvm

#.....Selecting best C and gamma.....
def BestParams(X,y,Xval,yval):
    y=y.ravel()        # Only to avoid Warnings
    C_list=[0.01,0.03,0.1,0.3,1,3,10,30]
    sigma_list=[0.01,0.03,0.1,0.3,1,3,10,30]
    least_error=100
    for C in C_list:
        for sigma in sigma_list:
            svm=skvm.SVC(C=C,kernel='rbf',gamma=(1.0/sigma))
            model=svm.fit(X,y)                   # Training with X and y

            prediction=model.predict(Xval)       # Choosing C and gamma using Xval and yval
            count=0
            for i in range(len(yval)):
                if prediction[i]!=yval[i]:
                    count+=1
            error=count/len(yval)        # mean
            if error<least_error:
                least_error=error
                best_C=C
                best_sigma=sigma
    print("    Best C : ",best_C)
    best_gamma=1.0/best_sigma
    print("Best gamma : ",best_gamma)            
    return best_C,best_gamma
