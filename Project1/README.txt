1: Decision Trees

DT_train_binary(X,Y,max_depth):

- First, define entropy function that calculates entropy of the full set.
- Loop across each feature in X, calculating entropy and subsequently information gain at 1st depth level.

It was here that I got stuck and, after speaking with you, shifted my focus to the other 2 problems.
This was still wildly informative for me as it made clear what were aspects of Python that I didn't know as well as I thought in comparison to something like MatLab (slicing, indexing, looping).
It also made clear what things I had little to no idea how to do or solve in Python (and more generally, from a CS perspective) such as recursion or recursive functions.
I am still struggling with the best way to iterate across branches of a decision tree, as well as construct a class or object in Python that facilitates this (much less understand the Python syntax that allows this).

DT_test_binary(X,Y,DT), DT_make_prediction(X,DT), DT_train_real(X,Y,max_depth), DT_test_real(X,Y,DT)

I didn't get to these as I focused on the other problems.

2: Nearest Neighbors

KNN_test(X_train,Y_train,X_test,Y_test,K):

This function follows the algorithm sketched out in lecture/textbook readings.

- First, calculate L2 norms between test sample and all training samples.
- Append those norms to a copy of the data, then sort by distance.
- Take the labels of the K-nearest training samples, sum them, then return the sign of the sum - which gives us the predicted label of the test sample.

We then define a second function, accuracy(X_test,Y_test,predicted_label):

- Set up variables used for accuracy calculation (true and false positive, true and false negatives).
- Run for loop that compares predicted labels to test labels, then bins each case into appropriate variable.
- Calculate and return accuracy based on following formula: (TP + TN)/(TP + TN + FP + FN).

choose_K(X_train,Y_train,X_val,Y_val)

I didn't get to finish this, so I will write up what I <think> would work to solve this.

- More or less, use KNN_test to solve for labels yielded by KNN method
- Loop through all possible K values - this will be 0 to N (number of data/sample points) - predict labels for test data based of each K.
- Write labels yielded by each K to a dictionary or numpy array.
- Once all possible K values have been iterated, use these labels to calculate accuracy for each K value using accuracy formula described above.

3: Clustering

K_means(X,K,mu):

Similar to 2., this also closely follows the algorithm laid out in the lecture and readings.

- First, check to see if user passes an argument for mu. If not, randomly initialize coordinates for K clusters.
- Compute L2 norms for each row in X to each cluster point.
- Extract the index of the lowest L2 norm - this is the cluster this row in X belongs to. If multiple are norms are equal, randomly select one (didn't implement this).
- Write these values to an array.

I also didn't finish this. The main feature missing is checking for cluster convergence, which I can imagine implementing by storing the calculated cluster value for the ith iteration, and comparing it with the cluster value generated at the ith+1 iteration.

K_means_better(X,K):

I didn't really get to this, and truthfully wasn't sure what was being asked here - particularly the number of iterations this function needed to run. At first read it just seems like this is convergence, but given "many times until the same set of cluster centers are returned a majority of the time" makes me wonder if this needs to run for some value of iterations (whether set by the user or some "optimal" iteration value).

