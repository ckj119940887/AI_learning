import numpy as np

from Deep_Learning.common.functions import softmax, cross_entropy_error


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

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        n = self.t.shape[0]
        # 标签是独热编码
        if self.t.size == self.y.size:
            dx = (self.y - self.t) * dout
        # 标签不是独热编码, 将预测值对应索引号的元素减1
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1

        return dx / n

