import numpy as np

#### Helper Function Block ####

def distance_point_to_hyperplane(pt, w, b):
    
    # return dist
    
    pass

def compute_margin(data, w, b):
       
    pass

def svm_test_brute(w, b, x):
    
    # return y
    
    pass

### Main Function ###

def svm_train_brute(data):
    
    ### FIND SUPPORT VECTORS ###
    
    posClass = []; negClass = []
    
    # Sort data into separate arrays based on label: posClass,negClass
    
    for i in range(len(data)):
        if data[i, -1] == 1:
            posClass = np.append(posClass, data[i, :])
        else:
            negClass = np.append(negClass, data[i, :])
    
    posClass = np.hsplit(posClass, (len(posClass)/ 3)); posClass = np.array(posClass) #Need to restructure 1d array into 2d array
    negClass = np.hsplit(negClass, (len(negClass)/ 3)); negClass = np.array(negClass)
    posClass = posClass[:, 0:-1]; negClass = negClass[:, 0:-1] #Remove labels from classes.
    
    ## Calculate all distances between positive and negative classes
    
    lenPos = len(posClass); lenNeg = len(negClass)
    
    posDist = np.zeros([lenPos, lenNeg])
    
    for i in range(len(posClass)):
        for j in range(len(negClass)):
            norm = np.linalg.norm(posClass[i] - negClass[j])
            posDist[j, i] = norm
            
    # Select positive and negative classified points w/ smallest distance - these will be support vectors

    SVIndex = np.where(posDist == np.amin(posDist))
    negClassIndex = SVIndex[0]; negClassIndex = SVIndex[0]
    posClassIndex = SVIndex[1]; posClassIndex = SVIndex[1]
    
    posPoint = posClass[posClassIndex]; negPoint = negClass[negClassIndex] #Place pos and neg support vector into own variable

    # What if 3 vectors?
    
    # posPoint = np.unique([posPoint]); negPoint = np.unique([negPoint]) #This breaks things horribly for some reason and I don't know why.

    ### CALCULATE MARGIN ###
    
    dir_w = posPoint - negPoint
    
    ### With 2 support vectors, gamma is half the distance between those two points.

    norm = np.linalg.norm(dir_w) #Find that distance,
    margin = norm / 2 #Then halve it. This gives you the margin (gamma).
    
    len_w = 1 / margin
    unit_dir_w = dir_w / len_w #Make direction unit length (length 1)
    
    ### Solve for weights and bias
    
    w = (unit_dir_w/norm) * margin
    b = (1 - 1 * (w[0] @ posPoint[0]))
    
    posPoint = np.append(posPoint, 1); negPoint = np.append(negPoint, -1)
    S = np.array([posPoint, negPoint])
    
    return w, b, S