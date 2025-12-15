import numpy as np

# 1.activation function
# 传入标量
def step_function0(x):
    if x >= 0:
        return 1
    else:
        return 0

# 传入向量或矩阵
def step_function(x):
    return np.array(x >= 0, dtype=int)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# RELU
def relu(x):
    return np.maximum(0, x)

# softmax
# 输入为向量
def softmax0(x):
    # prevent overflow
    x = x - np.max(x)
    e_x = np.exp(x) / np.sum(np.exp(x))
    return e_x

# 输入为矩阵
def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    e_x = np.exp(x) / np.sum(np.exp(x), axis=0)
    return e_x.T

# Identity
def identity(x):
    return x

# 2.loss function
# mean squared error
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# cross entropy error
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    # t is one-hot code format
    if y.size == t.size:
        t = t.argmax(axis=1)

    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n

# test
x = np.array([0, 1, 2, 3, 4, 5, -6, -7, -8, -9, -10])
# print(step_function(x))
# print(sigmoid(x))
# print(np.tanh(x))
# print(relu(x))