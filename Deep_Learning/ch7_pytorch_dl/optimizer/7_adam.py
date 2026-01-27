import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
W = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)

# define f(x1, x2) = 0.05 x1^2 + x2 ^ 2
def f(x):
    return W.dot(x**2)

# define parameter
lr = 0.1
n_iter = 500

# 更新参数x，并保存x的变化列表返回
def grad_desc(x, optimizer, n_iter):
    x_arr = x.detach().numpy().copy()
    for i in range (n_iter):
        y = f(x)
        y.backward()
        optimizer.step()
        optimizer.zero_grad()

        x_arr = np.vstack((x_arr, x.detach().numpy()))

    return x_arr

def adam(x, lr, betas, n_iters):
    x_arr = x.detach().numpy().copy()
    # 历史梯度累积
    h = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for i in range (n_iters):
        grad = 2 * x * W
        v = betas[0] * v + (1 - betas[0]) * grad
        h = betas[1] * h + (1 - betas[1]) * grad**2
        v_hat = v / (1 - betas[0] ** (i + 1))
        h_hat = h / (1 - betas[1] ** (i + 1))
        x.data -= lr * v_hat / (torch.sqrt(h_hat) + 1e-7)

        x_arr = np.vstack((x_arr, x.detach().numpy()))

    return x_arr

# 1. normal sgd
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
x_arr = grad_desc(X_clone, optimizer, n_iter)
plt.plot(x_arr[:, 0], x_arr[:, 1], color='red')

# 2. adam
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([X_clone], lr=lr, betas=(0.9, 0.999))
x_arr2 = grad_desc(X_clone, optimizer, n_iter)
plt.plot(x_arr2[:, 0], x_arr2[:, 1], color='blue')

# 3. manually implement adam
X_clone = X.clone().detach().requires_grad_(True)
x_arr3 = adam(X_clone, betas=(0.9, 0.999), lr=lr, n_iters=n_iter)
plt.plot(x_arr3[:, 0], x_arr3[:, 1], color='green', linestyle='--')

#
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
plt.legend(['SGD', 'adam', 'Manual'])

plt.show()
