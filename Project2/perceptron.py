import numpy as np

def perceptron_train(X, Y, maxIter=100):
    
    '''
    Parameters
    ----------
    X : n-dimensional numpy array
        N-dimensional feature array used for training weights and bias.
    Y : 1-dimensional numpy array
        Label vector used for training weights and bias.
    maxIter : int, optional
        Set the maximum number of iterations the algorithm iterates through. 
        The default is 100.

    Returns
    -------
    w : trained weights of the perceptron
    b : bias term of the perceptron

    '''
    
    b = 0 #initialize bias term at 0.
    w = np.zeros(X.shape[-1])

    for i in range(maxIter): #for each epoch,
        for i in range(len(X)): #iterate through each sample,
            a = np.dot(w, X[i]) + b #calculate activation
            
            if Y[i] * a > 0: #Weight update conditional
                continue
            else:
                b = b + Y[i] #Update bias term
                col = 0
                while col < len(w):
                    w[col] = w[col] + Y[i] * X[i, col]
                    col = col + 1
    return w, b

#Implement a function in python (perceptron_test) that takes testing data, the
#perceptron weights and bias as input and returns the accuracy on the testing
#data.

def perceptron_test(X_test, Y_test, w, b):
            
    '''
    Parameters
    ----------
    X_test : n-dim numpy array
        DESCRIPTION.
    Y_test : n-dim numpy array
        DESCRIPTION.
    w : 1-dim numpy array
        Trained weights for perceptron.
    b : int, float
        Trained bias value for perceptron.

    Returns
    -------
    accuracy : float
        Accuracy calculated from the predicted label over the actual label.
    '''

    Y_pred = []
    
    for i in range(len(X_test)):
        a = np.dot(w, X_test[i]) + b
        
        if Y_test[i] * a > 0:
            Y_pred.append(1)
        else:
            Y_pred.append(0)
            
    Y_pred = np.asarray(Y_pred)
    
    accuracy = sum(Y_pred)/len(Y_pred)
        
    return accuracy

