import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(3, 5)
        self.linear1.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
                [1.1, 1.2, 1.3, 1.4, 1.5]
            ]
        ).T
        self.linear1.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    def forward(self, x):
        x = self.linear1(x)
        return x

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, requires_grad=True)
t = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float, requires_grad=True)

model = Model()

y = model(x)

loss = nn.MSELoss()
loss_value = loss(y, t)

# backward
loss_value.backward()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()

for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])