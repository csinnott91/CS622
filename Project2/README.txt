1: Perceptron

perceptron_train(X, Y, maxIter=100):

We first initialize our weights and bias to be 0. Then, for a set number of epochs (maxIter), we iterate through each
sample and calculate the activation. Then, based off the activation and training label, we update (or don't update!)
weights and bias. We continue to do this until we reach the end of our set number of epochs, then return the weights
as a vector and our bias term as an integer.

perceptron_test(X_test, Y_test, w, b):

We first set an empty list to store our predicted labels in. We then compute activation based off our trained weights
and bias. Next, we check if the product of our test label and activation is greater than 0 - if it is, we store a 1
in our prediction label - otherwise we store a 0. Once we cycle through all of our test date, we calculate the accuracy
by summing the elements of our prediction label vector, and divide it by the vector's length.

2: Gradient Descent

gradient_descent(gradient_f, x_init = 5, eta = 0.01, maxIter = 52, threshold = 0.0001):

We first create a list, "x", using x_init as our first element. Then, we update x using our learning rate and gradient
of our function with the formula:

x_init = x_init - (eta * (gradient_f(x[i])))

We append this to our list, x, and continue iterating until we have completed maxIter number of epochs. Additionally,
we check step size against a threshold. If the step size is below this threshold, we terminate the loop and return
the minimum: otherwise, we continue until we drop below that threshold or reach the number of epochs specified.