import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import scipy.optimize as sco
import sklearn.svm as skvm

from CollaborativeFiltering_Cost import Collab_Filt_Cost
from Normalize_Ratings import Normalize
from GradientCheck import checkGradient
from Movie_List import get_Movie_List

#.....Loading and Visualizing Dataset.....
data1=scio.loadmat('ex8_movies.mat')
Y=data1['Y']
R=data1['R']
lam=0

r=np.where(R[0,:]==1)
print("\nAverage Rating of Movie 1 -Toy Story : ",np.mean(Y[0,r]))
plt.matshow(Y)
plt.xlabel('Users')
plt.ylabel('Movies')
#plt.show()

#.....Collaborative Filtering.....
    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data2=scio.loadmat('ex8_movieParams.mat')
X=data2['X']
theta=data2['Theta']
num_users=data2['num_users']
num_movies=data2['num_movies']
num_features=data2['num_features']

    # Reducing Dataset Size to run Faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
theta = theta[:num_users, :num_features]
params=np.hstack((X.flatten(),theta.flatten()))

Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

    # Checking Gradient and Cost Function
J,grad = Collab_Filt_Cost(params,Y,R,num_users,num_movies,num_features,lam)
print("\nCost of Dataset (lamda=0)   : ",J)
checkGradient(params,Y,R,num_users,num_movies,num_features,lam)

lam=1.5
J,grad = Collab_Filt_Cost(params,Y,R,num_users,num_movies,num_features,lam)
print("\nCost of Dataset (lamda=1.5) : ",J)
checkGradient(params,Y,R,num_users,num_movies,num_features,lam)

#.....My Ratings.....
X=data2['X']       # just  re-initializing
Y=data1['Y']
R=data1['R'].astype(bool)      # Convert R to bool to get 0/1 values
theta=data2['Theta']
num_users=data2['num_users']
num_movies=data2['num_movies']
num_features=data2['num_features']

my_ratings=np.zeros([X.shape[0],1])
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

Movie_List = get_Movie_List()      # loading Movie List

print("\nMy Ratings for movies :")
for i in range(len(my_ratings)):
    if my_ratings[i] !=0:
        print("Rated  %d" % my_ratings[i]," for movie ", Movie_List[i])

#.....TRAINING with New Movie Ratings using Scipy Optimizer.....
lam=10
Y=np.hstack((my_ratings,Y))      # added my_ratings to data
num_users= Y.shape[1]            # should be re-initialized because we added a new column
num_movies= X.shape[0]
num_features= X.shape[1]    

X= np.random.rand(num_movies,num_features)       # new randomly initialized X and theta with new user rating my_ratings
theta= np.random.rand(num_users,num_features)
params= np.hstack((X.flatten(),theta.flatten()))

R=np.hstack((my_ratings,R)).astype(bool)    
Y_norm,Y_mean= Normalize(Y,R)           # normalized data

print("\nTraining.........")
res= sco.minimize(Collab_Filt_Cost,params,args=(Y,R,num_users,num_movies,num_features,lam),method='CG',jac=True,options={'maxiter':100})
J= res.fun
X_and_theta= res.x

X= X_and_theta[:num_movies *num_features].reshape(num_movies,num_features)
theta= X_and_theta[num_movies *num_features:].reshape(num_users,num_features)
print("Cost after Training :",J)

#.....Recommendation of new Contents.....
pred= X.dot(theta.T)

pred_my_rating= pred[:,0] +Y_mean.flatten()

recom=np.argsort(-pred_my_rating)       # - sign is very imporatant to arrange it in descending order

print("\nRecommended Movies : ")
for i in range(10):
    j= recom[i]
    print("Predicting rating of  %1.2f" % pred_my_rating[j]," for movie ", Movie_List[j])