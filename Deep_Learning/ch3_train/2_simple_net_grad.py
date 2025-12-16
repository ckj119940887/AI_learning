import numpy as np
from Deep_Learning.common.gradient import numerical_diff
from Deep_Learning.common.functions import softmax,cross_entropy_error

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def forward(self,x):
        a = np.dot(x,self.W)
        z = softmax(a)
        return z

    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = SimpleNet()

    loss = lambda w: net.loss(x,t)

    # 计算梯度
    dW = numerical_diff(loss, net.W)

    print(dW)