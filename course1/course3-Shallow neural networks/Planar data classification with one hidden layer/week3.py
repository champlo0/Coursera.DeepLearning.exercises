# !/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :19/10/15 下午12:08
# !@Author  :CHAMPLOO
# !@File    :week3.py

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# set a seed so that the results are consistent
np.random.seed(1)

# load dataset
"""
X：一个numpy的矩阵，包含了这些数据点的数值
Y：一个numpy的向量，对应着的是X的标签【0 | 1】（红色:0 ， 蓝色 :1）
"""
X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plt.show()

# got size
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

"""
# 简单的逻辑回归模型 正确率47%（The dataset is not linearly separable, so logistic regression doesn't perform well.）
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y[0, :].T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y[0, :])
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")
"""


# Defining the neural network structure
def layer_sizes(X, Y):
    """
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）

    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
    """
    n_x = X.shape[0]  # 输入层
    n_h = 4  # ，隐藏层，硬编码为4
    n_y = Y.shape[0]  # 输出层

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y - 输出层节点的数量

    返回：
        parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言来确保数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }
    return parameters


"""
1.实施前向传播
2.计算损失
3.实现向后传播
4.更新参数（梯度下降）
5.合并到一个model中
"""


# 1.GRADED FUNCTION: forward_propagation 前向传播
def forward_propagation(X, parameters):
    """
    参数：
         X - 维度为（n_x，m）的输入数据。 (2,3)
         parameters - 初始化函数（initialize_parameters）的输出
    返回：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值 (1,3)
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
    """
    # get 参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 前向传播，计算并加入cache
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # 断言确定A2的shape
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# 2.GRADED FUNCTION: compute_cost 计算损失函数
def compute_cost(A2, Y, parameters):
    """
    计算方程（6）中给出的交叉熵成本，

    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值 (1,3)
         Y - "True"标签向量,维度为（1，数量）(1,3)
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量

    返回：
         成本 - 交叉熵成本给出方程（13）
    """

    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = - np.sum(logprobs) / m

    # E.g., turns [[17]] into 17 确保为数字
    cost = np.squeeze(cost)

    # 断言保证是float
    assert isinstance(cost, float)
    return cost


# 3.GRADED FUNCTION: backward_propagation 反向传播
def backward_propagation(parameters, cache, X, Y):
    """
    使用上述说明搭建反向传播函数。
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    m = X.shape[1]
    # 取回参数
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # 计算反向传播 dW1, db1, dW2, db2.
    dz2 = A2 - Y
    dW2 = np.dot(dz2, A1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.multiply(np.dot(W2.T, dz2), (1 - np.power(A1, 2)))
    dW1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# 4.GRADED FUNCTION: update_parameters 更新参数，梯度下降
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用上面给出的梯度下降更新规则更新参数

    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
    # 取出参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # 更新参数
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# 5.GRADED FUNCTION: nn_model 函数合并
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """
    np.random.seed(3)  # 设定随机种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)          # 1.前向传播
        cost = compute_cost(A2, Y, parameters)                  # 2.计算损失
        grads = backward_propagation(parameters, cache, X, Y)   # 3.反向传播
        parameters = update_parameters(parameters, grads)       # 4.更新参数
        if print_cost and i % 1000 == 0:
            print("第 ", i, " 次循环，cost为：" + str(cost))
    return parameters


# GRADED FUNCTION: predict 预测函数
def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions


"""
正式运行
"""
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# 打印准确率
predictions = predict(parameters, X)
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

"""
其他探索
"""
# 更改隐藏层的节点数量
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    plt.show()
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))

# 改变数据集
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

# 更改数据集 4个可选
dataset = "gaussian_quantiles"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y % 2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);

# 使用模型 Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
plt.title("Decision Boundary for hidden layer size " + str(4))
