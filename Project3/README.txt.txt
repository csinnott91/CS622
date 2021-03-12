1: PCA

compute_Z(X, centering=True, scaling=False):

We first check our centering and scaling arguments, and re-shape our data accordingly: subtracting the columnwise 
mean for each value when centering, and dividing each value by the stardard deviation if scaling. The result of this
is a matrix Z, which is just our input X with these changes applied to it.

compute_covariance_matrix(Z):

We then compute the covariance matrix of Z, which is simply Z times Z-transpose.

find_PCS(COV):

To find the eigenvalues and eigenvectors of COV, we use numpy's linalg.eig() function. We then use np.argsort to sort
our eigenvalues and eigenvectors corresponding to those eigenvalues in descending order.

def project_data(Z, PCS, L, k, var):

Finally, we then project our data in one of two ways: using k principle components, or using some number of principle
components with a cumulative variance greater than or equal to a value given with the var argument. We first check to
see if a k value greater than 0 has been passed. If it hasn't, we create an array with the cumulative sum of variance
explained by the eigenvalues (and subsequently, the principle components/eigenvectors corresponding to those eigenvalues).

We then iterate through that array and choose the number of values which have a cumulative variance greater than or 
equal to the argument var. Once we do that, we take the relevant principle components and use those to solve for our
projection Z-star. 

If k is greater than 0, then we immediately pull the k largest eigenvalues, and use their principle components to
calculate our projection Z-star.

2: SVM

Disclaimer: I worked on this for about ten hours before realizing I was taking the completely wrong approach to solve
this problem.

svm_train_brute(training_data):

My first task was to try and find support vectors, which I do through finding the two closest points of opposite class.
I split all data into positive and negative classes, then calculate distance for every combination of opposite labeled
points. From there, I take the points corresponding to the smallest distance and proceed. This worked well when there
were only two support vectors, but threw a fit when I had 3 equidistant points (as in test 4).

With those, I calculated the margin. This was largely framed from our last meeting; so I calculated the direction of w.
Since we have our support vectors already, we solve for the margin using the L2 norm of the direction vector of w
divided in half. We then make our direction vector unit length, and use that to backsolve for our weights and bias
before putting all of our previously identified support vectors into a numpy array.

This worked great for the first two test cases, started falling apart for test 3 right before the wheels came completely
off for test 4. I identified all of the support vectors correctly (though this didn't format correctly in test 4), but
couldn't calculate the correct weights and biases for 3 (and 4, but that's more of a result of how I coded this). I'm
still scratching my head as to why this didn't work for 3. I assume I screwed up the math somewhere, but I'm not sure
that's the case given my correct output for tests 1 and 2.

svm_train_multiclass(training_data):

I didn't get to this. I'm don't think my approach for the binary problem would work for this at ALL (it didn't really
work for the binary problem, after all). 