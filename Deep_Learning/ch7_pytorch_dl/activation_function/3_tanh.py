import torch
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

# define x and y
x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = torch.tanh(x)

# create sub plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# sigmoid
# 尽量用detach来获得数据，而不要直接使用x.data, y.data
ax[0].plot(x.detach(), y.detach(), color='purple')
ax[0].set_title('tanh(x)')
ax[0].axhline(y=1, color='gray', alpha = 0.5, linewidth=1)
ax[0].axhline(y=-1, color='gray', alpha = 0.5, linewidth=1)

# sigmoid(x)'
# 反向传播计算梯度，需要先得到一个标量
y.sum().backward()

ax[1].plot(x.data, x.grad, color='purple')
ax[1].set_title('tanh(x)\'')

plt.show()
