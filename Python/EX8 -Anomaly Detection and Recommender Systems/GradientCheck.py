import numpy as np

from CollaborativeFiltering_Cost import Collab_Filt_Cost

def checkGradient(params, Y, R, num_users, num_movies, num_features, lam):
    
    print('\nComputed Gradient \t Gradient from () \t Difference')

    myeps = 0.0001
    nparams = len(params)
    epsvec = np.zeros(nparams)

    mygrads = Collab_Filt_Cost(params,Y,R,num_users,num_movies,num_features,lam)[1]
    
    for _ in range(10):
        idx = np.random.randint(0,nparams)
        epsvec[idx] = myeps                 # only that position(idx) should be tampered by epsilon
        loss1 = Collab_Filt_Cost(params-epsvec,Y,R,num_users,num_movies,num_features,lam)[0]
        loss2 = Collab_Filt_Cost(params+epsvec,Y,R,num_users,num_movies,num_features,lam)[0]
        mygrad = (loss2 - loss1) / (2*myeps)
        epsvec[idx] = 0
        print('%0.15f \t %0.15f \t %0.15f' % (mygrad, mygrads[idx],mygrad - mygrads[idx]))