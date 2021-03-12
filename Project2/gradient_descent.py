def gradient_descent(gradient_f, x_init = 5,
                     eta = 0.01, maxIter = 52,
                     threshold = 0.0001):
    '''

    Parameters
    ----------
    gradient_f : TYPE
        Gradient of the function to be descended.
    x_init : int, float; optional
        Initial point on x-axis on function to begin gradient descent. 
        The default is 5.
    eta : int, float; optional
        Learning rate. The default is 0.01.
    maxIter : int, optional
        Set number of steps for gradient descent to be performed. 
        The default is 52.
    threshold : float, optional
        Set the threshold for convergence.
        The default is 0.0001.

    Returns
    -------
    x : X-value for global minimum of function.

    '''

    x = [x_init]
    for i in range(maxIter): 
        x_init = x_init - (eta * (gradient_f(x[i])))
        x.append(x_init)
        
        diff = abs(x_init - x[-2])
        if any(diff >= threshold):
            continue
        else:
            break
    
    return x[-1]
