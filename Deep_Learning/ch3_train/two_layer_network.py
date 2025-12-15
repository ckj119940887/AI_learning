import numpy as np

import os
import sys

from Deep_Learning.common.functions import sigmoid, softmax, cross_entropy_error

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from common.gradient import numerical_gradient
from common.functions import *

class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self,x):
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
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
