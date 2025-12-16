import numpy as np
from collections import OrderedDict

from Deep_Learning.common.functions import sigmoid, softmax, cross_entropy_error
from Deep_Learning.common.gradient import numerical_gradient
from Deep_Learning.common.layers import *

class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 按照顺序定义层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def forward(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.forward(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # 预测
        y = self.forward(x)
        # 将概率转换为序号
        y = np.argmax(y, axis=1)
        # 计算准确度
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        print(f"x: {type(x)}, t: {type(t)}")
        loss = lambda w: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss, self.params['W1'])
        grads['b1'] = numerical_gradient(loss, self.params['b1'])
        grads['W2'] = numerical_gradient(loss, self.params['W2'])
        grads['b2'] = numerical_gradient(loss, self.params['b2'])

        return grads

    # 利用反向传播实现梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # last layer
        dout = 1
        dout = self.lastLayer.backward(dout)

        # backward,反向遍历
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 提取Affine 层的参数梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
