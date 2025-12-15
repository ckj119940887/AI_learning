import numpy as np
from two_layer_network import TwoLayerNet
import os, sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from common.load_data import *

# 加载数据
x_train, x_test, t_train, t_test = get_data()

# model
model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

batch_size = 100
iter_per_epoch = x_train.shape[0] // batch_size

iters_num = 1000
learning_rate = 0.1

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for i in range(iters_num):
    # 随机选取batch_size个训练数据
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(type(x_batch))
    print(type(t_batch))

    # 计算当前的梯度
    grad = model.numerical_gradient(x_batch, t_batch)
    for key in ['W1', 'W2', 'b1', 'b2']:
        model.params[key] -= learning_rate * grad[key]

    loss = model.loss(x_batch, t_batch)
    train_loss.append(loss)

    if i % iter_per_epoch == 0:
        train_acc.append(model.accuracy(x_train, t_train))
        test_acc.append(model.accuracy(x_test, t_test))
        train_acc.append(train_acc)
        test_acc.append(test_acc)
        print(f"train loss : {train_loss[i]}, train acc : {train_acc[i]}, test acc : {test_acc[i]}")