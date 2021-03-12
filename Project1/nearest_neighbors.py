#nearest_neighbors.py
import numpy as np
import scipy.spatial as ss

#Implement a function in python that takes training data, test data, and K as inputs. 
#The function should return the accuracy on the test data. 
#The training data and test data should have the same format as described earlier for the decision tree problems. 
#Your function should be able to handle any dimension feature vector, with real-valued features. 
#Remember your labels are binary, and they should be -1 for the negative class and 1 for the postive class.

def KNN_test(X_train,Y_train,X_test,Y_test,K):

    distance = []; i = 0; row = 0
    
    while i <= len(X_train):     #1. find distance from test sample to all training samples
        distance = np.append(distance, (ss.distance.euclidean(X_train[row], X_test[row]))) #Calculate L2 norm, write to vector
        i = i + 1

    sorted_vector = np.append(Y_train, distance) #2 Append L2 vector to a copy of the rest of data, 
    sorted_vector = np.argsort(sorted_vector) #then sort by distance.

    while i <= K:

            vote_total = sum(sorted_vector[2,0:K]) #3. Predict label of test sample by sign of sum of K-nearest training samples.
            predicted_label = np.sign(vote_total)
            
            i = i + 1
            
            return predicted_label
        
def accuracy(X_test,Y_test,predicted_label):
    
    TP = 0; TN = 0; FP = 0; FN = 0
    
    for i in X_test[:, 0]: #Loop for boolean operators to compare predicted label to actual test label, and sort into appropriate variable.
        if predicted_label(i) == 1:
            if predicted_label(i) == Y_test(i):
                TP = TP + 1
            else:
                FP = FP + 1
        if predicted_label(i) == -1:
            if predicted_label(i) == Y_test(i):
                TN = TN + 1
            else:
                FN = FN + 1
                
    acc = (TP + TN)/(TP + TN + FP + FN)
    
    return acc

#622 Implement the following function in python:

def choose_K(X_train,Y_train,X_val,Y_val):

    for i in range (X_val[:, 0]):
        distance[i, 0] = np.linalg.norm(X_val)
    return K

#Decide K based on the validation data

# REMEMBER - K CANNOT BE GREATER THAN N


#that takes training data and validation data as inputs and returns a K value. This function
#must iterate through all possible K values and choose the best K for the given training data
#and validation data. K must be chosen in order to achieve the best accuracy on the validation
#data.