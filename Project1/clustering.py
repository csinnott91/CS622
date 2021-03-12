#clustering.py
import numpy as np
import scipy.spatial as ss
import random as random

def K_Means(X,K,mu):

#that takes feature vectors X and a K value as input, and returns a numpy array of cluster centers C.
# Your function should be able to handle any dimension of feature vectors and any K > 0.
#mu is an array of initial cluster centers, with either K or 0 rows. If mu is empty, then you must
#initialize the cluster centers randomly. Otherwise, start with the given cluster centers.

    X = X.astype(int)
    if K <= 0: #Check for valid hyperparameter selection.
        print("Please select a K value greater than 0.")

#Algorithm steps:
    

#1. Initialize cluster centers randomly

    if mu == np.array([]): #Check if user has input a value or values for mu,
        for i in K:
            mu[i, i] = [random.randint, random.randint] #and if they haven't, randomly generate 2d coordinates for a number of cluster centers = K.

#2. Compute distance from each point to each cluster center - see Hand's lecture "Intro to ML Clustering" ~3:50 mark.
### We need to iterate over this to reach convergence.

    mu_distance = np.array([])

    for i in mu: #For each cluster center,
        for i in X: #and for each row in X,
            mu_distance[i, 0] = ss.distance.euclidean(X[i, :], mu[i, :]) #Calculate L2 norm relative to each cluster point,

#3. Assign each point to a cluster using minimum distance

    for i in X[i, -1:-K]: #For every row, but only the L2 norm columns of each X
        X[i, -1] = np.argmin(np.min(X, axis=-0)) #Extract the index of the lowest L2 norm - the cluster the row belongs to - and append it at end.
        
        if X[i,-1:K] == True: #if multiple L2 norms are equal,
            pass #randomly select one, else proceed as usual

            #Calculate C for each cluster of points

    C = np.array([])

    for i in X: #For every row,
        cluster_values = np.where(X[:, -1] == i)  #Calculate the mean based off values sharing a cluster value.
        C[i:] = np.mean(cluster_values)
        
    for i in C: #Check to see if cluster centers have converged.
        if C == mu:
            break
        else:
            continue

    return C

#622 Implementation

def K_Means_better(X,K):
    C = np.array([])
    
    for i in len(K):
        mu = np.array([random.randint, random.randint])
    
    mu = [random.randint(K), random.randint(K)]
    
    
    
    while C != mu:
        K_means(X, K, mu)
    else:
        return C
    

#that takes feature vectors X and a K value as input and returns a numpy array of cluster centers C.
#Your function should be able to handle any dimension of feature vectors and any K > 0.
#Your function will run the above-implemented K_Means function many times until the same set of
#cluster centers are returned a majority of the time. At this point, you will know that those cluster
#centers are likely the best ones. K_Means_Better will return those cluster centers.
