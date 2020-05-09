import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

#%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)     # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y, dsplot = load_dataset()
dsplot.show()


# classifier to separate the blue dots from the red dots
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.

    Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations to run gradients descent
        print_cost -- if True, print the cost every 1000 iterations
        initialization -- flag to choose which initialization to use ("zeros", "random" or "he")

    Returns:
        parameters -- parameters learnt by the model
    """

    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  # number of examples
    layer_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)

    # Loop (gradients descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)

        # Update parameters.
        paremeters = update_parameters(parameters, grads, learning_rate)

        # Print the Loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# Zero initialization -- this does not work well since it fails to "break symmetry"
def initialize_parameters_zeros(layers_dims):
        """
        Arguments:
            layers_dims -- python array (list) containing the size of each layer.

        Returns:
            parameters -- python dictionary containing our parameters "W1", "b1", ...., "WL", "bL":
                W1 -- weight matrix of shape (layer_dims[1], layer_dims[0])
                b1 -- bias vector of shape (layer_dims[1], 1)
                ...
                WL -- weight matrix of shape (layer_dims[L], layer_dims[L-1])
                bL -- bias vector of shape (layer_shape[L], 1)
        """

        parameters = {}
        L = len(layers_dims)            # number of layers in the network

        for l in range(1, L):
            parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters

parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization="zeros")
print("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the training set:")
predictions_test = predict(test_X, test_Y, parameters)

""" The performance is really bad, and the cost does not really decrease, and the algorithm performs 
    no better than random guessing. Why? Lets look at the details of the predictions and the decision boundary:"""
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
""" The model is predicting 0 for every example.
      In general, initializing all the weights to zero results in the network failing to break symmetry. 
      This means that every neuron in each layer will learn the same thing, and you might as well 
      be training a neural network with  n[l]=1n[l]=1  for every layer, and the network is 
      no more powerful than a linear classifier such as logistic regression.
    
    What you should remember:
    - The weights  W[l]W[l]  should be initialized randomly to break symmetry.
    - It is however okay to initialize the biases  b[l]b[l]  to zeros. Symmetry is still broken 
      so long as  W[l]W[l]  is initialized randomly."""


# Random initialization -- To break symmetry, lets intialize the weights randomly. Following random initialization,
# each neuron can then proceed to learn a different function of its inputs.
#   Exercise: Implement the following function to initialize your weights to large random values (scaled by *10)
#       and your biases to zeros. Use np.random.randn(..,..) * 10 for weights and np.zeros((.., ..)) for biases.
#       We are using a fixed np.random.seed(..) to make sure your "random" weights match ours, so don't worry
#       if running several times your code gives you always the same initial values for the parameters.
def initialize_parameters_random(layer_dims):
    """
    Arguments:
        layer_dims -- python array (list) containing the size of each layer.
    Returns:
        parameters -- python dictionary containing our parameters "W1", "b1", ..., "WL", "bL":
            W1 -- weight matrix of shape (layer_dims[1], layer_dims[0])
            b1 -- bias vector of shape (layer_dims[1], 1)
            ...
            WL -- weight martix of shape (layer_dims[L], layer_dims[L-1])
            bL -- bias vector of shape (layer_dims[L], 1)
    """
    np.random.seed(3)               # this makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layer_dims)             # integer representing the number of layers

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization="random")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

# If you see "inf" as the cost after the iteration 0, this is because of numerical roundoff;
# a more numerically sophisticated implementation would fix this. But this isn't worth worrying about for our purposes.

# It looks like you have broken symmetry, and this gives better results. than before.
#   The model is no longer outputting all 0s.
print(predictions_train)
print(predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

""" Observations:
    - The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) 
        outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong 
        it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
    - Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
    - If you train this network longer you will see better results, but initializing with overly large random numbers 
        slows down the optimization.
    - Initializing weights to very large random values does not work well.
    - Hopefully intializing with small random values does better. The important question is: 
        how small should be these random values be? Lets find out in the next part! """


# He initialization -- his is similar except Xavier initialization uses a scaling factor for the weights
            # W[l] of sqrt(1/layers_dims[l-1]) where He initialization would use sqrt(2/layers_dims[l-1])
""" Hint: This function is similar to the previous initialize_parameters_random(...). 
        The only difference is that instead of multiplying np.random.randn(..,..) by 10, you will multiply it 
        by sqrt(2/{dimension of the previous layer}), which is what He initialization recommends 
        for layers with a ReLU activation.  """
def initialize_parameters_he(layer_dims):
    """
    Arguments:
        layer_dims -- python array (list) containing the size of each layer.
    Returns:
        parameters -- python dictionary containing out parameters "W1", "b1", ..., "WL", "bL":
            W1 -- weight matrix of shape (layer_dims[1], layer_dims[0])
            b1 -- bias vector of shape (layer_dims[1], 1)
            ...
            WL -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
            bL -- weight matrix of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)       # integer representing the number of layers

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b1 = " + str(parameters["b2"]))


parameters = model(train_X, train_Y, initialization="he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set")
predictions_test = predict(test_X, test_Y, parameters)


plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# Observations:
#   The model with He initialization separates the blue and the red dots very well in a small number of iterations.

""" Conclusions:
        You have seen three different types of initializations. 
        For the same number of iterations and same hyperparameters the comparison is:

        **Model**	                                    **Train accuracy**	            **Problem/Comment**
        3-layer NN with zeros initialization	                50%	                    - fails to break symmetry
        3-layer NN with large random initialization	            83%	                    - too large weights
        3-layer NN with He initialization	                    99%	                    - recommended method

Random initialization is used to break symmetry and make sure different hidden units can learn different things
Don't intialize to values that are too large """

""" He initialization works well for networks with ReLU activations."""
