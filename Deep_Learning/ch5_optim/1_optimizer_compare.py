import numpy as np
import matplotlib.pyplot as plt
from Deep_Learning.common.optimizer import *
from collections import OrderedDict

def f(x, y):
    return x**2 / 20.0 + y**2

# 梯度函数
def fgrad(x, y):
    return x / 10.0, 2.0 * y

# initial value
init_point = (-7.0, 2.0)

params = {}
params['x'] = init_point[0]
params['y'] = init_point[1]

grads = {}
grads['x'] = 0.0
grads['y'] = 0.0

# 定义优化器
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.9)
optimizers['Momentum'] = Momentum(lr=0.1, momentum=0.85)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

iter_num = 30
idx = 1

# 遍历优化器，调用update方法
#
for key in optimizers:
    optimizer = optimizers[key]

    x_history, y_history = [], []
    params['x'] = init_point[0]
    params['y'] = init_point[1]

    for i in range(iter_num):
        x_history.append(params['x'])
        y_history.append(params['y'])

        # 计算梯度
        grads['x'], grads['y'] = fgrad(params['x'], params['y'])
        # 调用update方法更新
        optimizer.update(params, grads)

    # figure
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color='blue', label=key, markersize=2)
    plt.xlim([-10, 10])
    plt.ylim([-5, 5])
    plt.plot(0, 0, '+')
    # 画等高线
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0
    plt.contour(X, Y, Z)
    plt.legend(loc='best')

plt.show()