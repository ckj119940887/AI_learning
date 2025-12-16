import numpy as np
import math
import matplotlib.pyplot as plt
from Deep_Learning.common.functions import *
from Deep_Learning.common.layers import *
from Deep_Learning.common.load_data import *
from collections import OrderedDict # 有序字典
from two_layer_network import *

# 加载数据
x_train, x_test, t_train, t_test = get_data()

# model
model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = math.ceil(train_size / batch_size)

iters_num = 10000
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    # 随机选取batch_size个训练数据
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算当前的梯度
    grad = model.gradient(x_batch, t_batch)
    for key in ['W1', 'W2', 'b1', 'b2']:
        model.params[key] -= learning_rate * grad[key]

    loss = model.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = model.accuracy(x_train, t_train)
        test_acc = model.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train loss : {loss}, train acc : {train_acc}, test acc : {test_acc}")

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle='--')
plt.legend()
plt.ylim(0, 1.0)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()