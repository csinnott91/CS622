import numpy as np
import math as m
import random.randint

def DT_train_binary(X, Y, max_depth):
    
    def entropy(X,Y):
        p_obs = sum(Y) / len(Y)
        p_obs_inv = 1 - p_obs
        entropy = -p_obs * m.log(p_obs, 2) - p_obs_inv * m.log(p_obs_inv, 2)
        return entropy
    
    for i in X[:, i]:
        
        entropy_full = entropy(X,Y)
        entropy_subset_0 = entropy(X[:, i], Y) for Y == 0 or Y == -1
        entropy_subset_1 = entropy(X[:, i], Y) for Y == 1
        p_obs_0 = sum(Y) / len(Y) for Y == 0
        p_obs_1 = sum(Y) / len(Y) for Y == 1
        IG = entropy_full - sum(p_obs_0*entropy_subset_0, p_obs_1*entropy_subset_1)
        
        full_data = np.append(X, Y, IG[-1, i]

    
    
    
        
    
    
    
    def IG(X,Y,Xi,Yi):

        IG = entropy - sum(entropy(X[],Y[])) for i in X
        
        return IG
    
    return DT


def DT_test_binary(X,Y,DT):
    pass
    DT = "Unfinished!"
    return DT

def DT_make_prediction(X,DT):
    pass
    DT = "Unfinished!"
    return DT

def DT_train_real(X,Y,max_depth):
    pass
    accuracy = "Unfinished!"
    return accuracy

def DT_test_real(X,Y,DT):
    pass
    accuracy = "Unfinished!"
    return accuracy