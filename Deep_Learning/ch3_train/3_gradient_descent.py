import numpy as np
import matplotlib.pyplot as plt

from Deep_Learning.common.gradient import numerical_gradient


# 梯度下降
def gradient_descent(f, init_x, lr=0.01, num_iter=100):
    x = init_x
    x_history = []
    for i in range(num_iter):
        grad = numerical_gradient(f, x)

        # 保存当前点到列表
        x_history.append(x.copy())
        x = x - lr * grad

    return x, np.array(x_history)

# test, f(x1, x2) = x1 ^ 2 + x2 ^ 2
def f(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    num_iter = 20

    x, x_history = gradient_descent(f, init_x, lr)

    plt.plot(x_history[:, 0], x_history[:, 1])
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5],  '--b')
    plt.xlim([-4, 4])
    plt.ylim([-4.5, 4.5])
    plt.show()