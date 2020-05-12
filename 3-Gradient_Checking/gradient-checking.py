""" Final assignment for this week! In this assignment you will learn to implement and use gradient checking.

    You are part of a team working to make mobile payments available globally, and are asked to build
    a deep learning model to detect fraud--whenever someone makes a payment, you want to see
    if the payment might be fraudulent, such as if the user's account has been taken over by a hacker.

    But backpropagation is quite challenging to implement, and sometimes has bugs.
    Because this is a mission-critical application, your company's CEO wants to be really certain that
    your implementation of backpropagation is correct. Your CEO says, "Give me a proof that
    your backpropagation is actually working!" To give this reassurance, you are going to use "gradient checking".

    Let's do it!"""

import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def forward_propagation(x, theta):
    """
    Implement the Linear forward propagation (compute J -- J(theta) = theta * x)

    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well

    Returns:
        J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    J = x * theta
    return J


x, theta = 2, 4
J = forward_propagation(x, theta)
print("J = " + str(J))


# Implement the backward propagation step (derivative computation) -- the derivative of J(theta) = theta*x
# with respect to theta.You should get dtheta = partial J / partial theta = x

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta
    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well

    Returns:
        dtheta -- the gradient of the cost with respect to theta
    """
    dtheta = x
    return dtheta

x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print("dtheta = " + str(dtheta))



""" Gradient check:
    First compute "gradapprox" 
    Then compute the gradient using backward propagation, and store the result in a variable "grad"
    Finally, compute the relative difference between "gradapprox" and the "grad"
    
    You will need 3 Steps to compute this formula:
        - compute the numerator using np.linalg.norm(...)
        - compute the denominator. You will need to call np.linalg.norm(...) twice.
        - divide them.
    If this difference is small (say less than 10^{-7}), you can be quite confident that you have computed 
    your gradient correctly. Otherwise, there may be a mistake in the gradient computation."""

def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward prop

    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        epsilon -- tiny shift to the input to compute approximated gradient

    Returns:
        difference -- difference between the approximated gradient and the backward propagation gradient
    """

    # compute the "gradapprox". epsilon is small enough, no need to worry about limit
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = thetaplus * x
    J_minus = thetaminus * x
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = x

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")

    return difference

x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))
# difference = 2.919335883291695e-10 --> since the difference is smaller than the 10^{-7} threshold, gradient is correct

""" In the more general case, your cost function J has more than a single 1D input. 
        When you are training a neural network, theta actually consists of multiple matrices W[l] and biases b[l]! 
        It is important to know how to do a gradient check with higher-dimensional inputs. Let's do it!"""

""" N-dimensional gradient checking
        Let's look at your implementations for forward propagation and backward propagation."""

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost)

    Arguments:
        X -- training set for m examples
        Y -- true "labels" for m examples
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (5, 4)
                        b1 -- bias vector of shape (5, 1)
                        W2 -- weight matrix of shape (3, 5)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)

    Returns:
        cost -- the cost function (logistic cost for one example)
    """
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1/m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache

def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation

    Args:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()

    Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter,
                        activation and pre-activation variables.
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
#   dW2 = 1./m * np.dot(dZ2, A1.T) * 2
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
#   db1 = 4./m * np.sum(dZ1, axis=1, keepdims=True)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

# You obtained some results on the fraud detection test set but you are not 100% sure of your model.
# Nobody's perfect! Let's implement gradient checking to verify if your gradients are correct.

#  Want to compare "gradapprox" to the gradient computed by backpropagation

""" However, theta is not a scalar anymore. It is a dictionary called "parameters". We implemented 
    a function "dictionary_to_vector()" for you. It converts the "parameters" dictionary into a vector called "values", 
    obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them.

    The inverse function is "vector_to_dictionary" which outputs back the "parameters" dictionary.
    We have also converted the "gradients" dictionary into a vector "grad" using gradients_to_vector(). 
    You don't need to worry about that.
    
    To compute J_plus[i]:
        Set theta^{+} to np.copy(parameters_values)
        Set theta[i]^{+} to theta[i]^{+} + epsilon
        Calculate J[i]^{+} using to forward_propagation_n(x, y, vector_to_dictionary(theta^{+})).
        To compute J_minus[i]: do the same thing with theta^{-}
        Compute gradapprox[i] = J[i]^{+} - J[i]^{-} / (2 * epsilon)
        Thus, you get a vector gradapprox, where gradapprox[i] is an approximation of the gradient 
        with respect to parameter_values[i]. You can now compare this gradapprox vector to the gradients vector 
        from backpropagation. Just like for the 1D case (Steps 1', 2', 3'), 
        compute: difference = |grad - gradapprox|_2} / | grad |_2 + | gradapprox |_2 """

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
        parameters -- python dictionary containing your parameters
        grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameter s.
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient

    Returns:
        difference -- difference between approximated gradient and the backward propagation gradient
    """
    # Set up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon".  Output: "J_plus[i]"
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon".     Output: "J_minus[i]".
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(gradapprox - grad)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator / denominator

    if difference > 1.2e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) +
              "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) +
              "\033[0m")
    return difference


X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)

""" It seems that there were errors in the backward_propagation_n code we gave you! 
    Good that you've implemented the gradient check. Go back to backward_propagation and try to find/correct the errors 
    (Hint: check dW2 and db1).
    
    Rerun the gradient check when you think you've fixed it. Remember you'll need to re-execute the cell 
    defining backward_propagation_n() if you modify the code.

    Can you get gradient check to declare your derivative computation correct? 
    Even though this part of the assignment isn't graded, we strongly urge you to try to find the bug and 
    re-run gradient check until you're convinced backprop is now correctly implemented."""

""" Note

Gradient Checking is slow! Approximating the gradient with 
partial J / partial theta approx=  J(theta + epsilon) - J(theta - epsilon) / {2 * epsilon} is computationally costly. 
For this reason, we don't run gradient checking at every iteration during training. 
Just a few times to check if the gradient is correct.

Gradient Checking, at least as we've presented it, doesn't work with dropout. 
You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, 
then add dropout.
Congrats, you can be confident that your deep learning model for fraud detection is working correctly! 
You can even use this to convince your CEO. :)

Gradient checking verifies closeness between the gradients from backpropagation and 
the numerical approximation of the gradient (computed using forward propagation).
Gradient checking is slow, so we don't run it in every iteration of training. 
You would usually run it only to make sure your code is correct, then turn it off and use backprop 
for the actual learning process."""
