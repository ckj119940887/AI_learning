import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
W = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)

# define f(x1, x2) = 0.05 x1^2 + x2 ^ 2
def f(x):
    return W.dot(x**2)

# define parameter
lr = 0.8
n_iter = 500


# 1. define optimizer
optimizer = torch.optim.SGD([X], lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# update X
x_arr = X.detach().numpy().copy()
lr_list = []
for i in range (n_iter):
    y = f(X)
    y.backward()
    optimizer.step()
    optimizer.zero_grad()

    x_arr = np.vstack((x_arr, X.detach().numpy()))

    lr_list.append(optimizer.param_groups[0]['lr'])

    # 学习率衰减
    scheduler.step()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
ax[0].contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
ax[0].plot(x_arr[:, 0], x_arr[:, 1], color='red')
ax[0].set_title('gradient descent')

ax[1].plot(lr_list, 'k')
ax[1].set_title('learning rate')

plt.show()
