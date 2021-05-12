import math
import numpy as np

# Sets the initial weights to zero
def set_init(size):
    weights = []

    for i in range(size):
        weights.append(0)

    return weights


# Takes in one sample and updates the weights base on it
def update_weight(X, Y, weights, size):
    # w =  w + xy
    for i in range(size):
        weights[i] = weights[i] + (X[i] * Y)

    return weights


# Takes in the previous bias and updates it based in the label of the sample
def update_bias(bias, Y):
    # b = b + y
    return bias + Y


# Takes in the activation and multiplies it by the label of the sample
def check_update(activation, Y):
    return activation * Y


# Calculates the activation based on the weights and X values
def get_activation(X, weights, size, bias):
    # sum x_i * w_i + b
    a_value = 0

    for i in range(size):
        a_value += weights[i] * X[i]

    return a_value + bias


# Takes the final weights and bias and stores them in a list
def append_weights_bias(weights, bias):
    # weight_bias[0] has list of weights [1] has the bias
    weight_bias = []
    weight_bias.append(weights)
    weight_bias.append(bias)

    return weight_bias


def perceptron_train(X,Y):

    num_of_samples, num_of_features = np.shape(X)
    bias = 0
    num_non_updates = 0
    epoch = 1
    weights = set_init(num_of_features)  # sets initial weights

    while num_non_updates < num_of_samples:  # checks if it runs a full epoch without updates

        for i in range(num_of_samples):

            activation = get_activation(X[i], weights, num_of_features, bias)

            if check_update(activation, Y[i][0]) <= 0:  # checks if it needs to update
                weights = update_weight(X[i], Y[i][0], weights, num_of_features)
                bias = update_bias(bias, Y[i][0])
                num_non_updates = 0

            elif epoch > 100:  # stops at a hundred epochs
                num_non_updates = num_of_samples

            else:
                num_non_updates += 1

        epoch += 1

    return append_weights_bias(weights, bias)  # puts the weight and bias into a list


def perceptron_test(X_test, Y_test, w, b):

    num_of_samples, num_of_features = np.shape(X_test)
    correct = 0

    for i in range(num_of_samples):
        activation = get_activation(X_test[i], w, num_of_features, b)
        if check_update(activation, Y_test[i][0]) > 0:  # Checks if it is correct
            correct += 1

    return correct / num_of_samples  # returns the accuracy
