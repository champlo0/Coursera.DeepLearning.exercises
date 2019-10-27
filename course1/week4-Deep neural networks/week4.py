# !/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :19/10/24 下午6:08
# !@Author  :CHAMPLOO
# !@File    :week4.py


import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_app_utils_v2 import *
import time
import scipy
# from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load_ext autoreload
# autoreload 2

np.random.seed(1)

"""
1.  前向传播（保存cache）--> AL（yhat）
1.1 线性前向：z = WA + b
1.2 激活函数：L-1（linear+relu）+ linear+sigmoid
2.  计算cost
3.  后向传播 Y，yhat，caches --> grads（字典）
3.1 线性后向 dZ --> dA，dW，db
3.2 后向激活
4.  更新参数 w = w - rate*dw ； b = b - rate*db

"""


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化参数
    :param n_x:INPUT 层数
    :param n_h:隐藏 层数
    :param n_y:OUTPUT 层数
    :return:
    parameters：初始化元素的字典：
            W1：大小(n_h, n_x)
            b1：大小(n_h, 1)
            W2：大小(n_y, n_h)
            b2：大小(n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: 维度 如【2，4，1】INPUT：2 HIDDEN：4 OUTPUT：1
    :return:初始化好的参数 W & B
    """

    np.random.seed(3)
    parameters = {}

    for i in range(1, len(layer_dims)):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    """
    线性前向传播
    :param A: 上一层的activation，或INPUT X(上层size， 样本数)
    :param W: （本层size， 上层size）
    :param b: （本层size， 1）
    :return: Z：（本层size，样本数）
            cache：A，W，b
    """
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    线性前向激活函数
    :param A_prev: 上一层的activation，或INPUT X(上层size， 样本数)
    :param W: （本层size， 上层size）
    :param b: （本层size， 1）
    :param activation: "sigmoid" or "relu"
    :return:
    A: post-activation value
    cache：一个包含了"linear_cache" and "activation_cache"的字典
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    L层前向传播
    :param X: (input size, 样例number）
    :param parameters:output of initialize_parameters_deep()
    :return:
    AL：最后输出的yhat (1, 样本数）
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 层数
    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def compute_cost(AL, Y):
    """
    计算cost函数
    :param AL: shape (1, number of examples)
    :param Y:  shape (1, number of examples)
    :return: cost
    """
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y)))

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    线性后向传播
    :param dZ:Gradient of the cost
    :param cache:tuple of values (A_prev, W, b)
    :return:
    dA_prev -- Gradient of the cost shape == A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (db.shape == b.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    线性后向激活函数
    :param dA: 当前层的dA
    :param cache:当前层的tuple of values (linear_cache, activation_cache)
    :param activation:"sigmoid" or "relu"
    :return:
    dA_prev -- Gradient of the cost shape == A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    L层后向传播
    :param AL: output of the forward propagation---yhat
    :param Y: Y
    :param caches:list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    :return:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] \
        = linear_activation_backward(dAL, caches[-1], activation="sigmoid")

    # lth layer: (RELU -> LINEAR) gradients.
    for l in reversed(range(L - 1)):
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(
            grads["dA" + str(l + 2)], caches[l], activation="relu")
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    更新参数
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # 层数
    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return parameters


# Application 识别猫

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        ### END CODE HERE ###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
