import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = x / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        dx = dout * self.y * (1.0 - self.y)
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.X_original_shape = None # 原始X张量的形状
        # 保存参数的梯度
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X_original_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        Y = np.dot(self.X, self.W) + self.b
        return Y

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)

        return dX.reshape(*self.X_original_shape)
