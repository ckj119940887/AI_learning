import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

X = torch.randn(100, 1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
noise = torch.randn(100, 1) * 0.5
y = w * X + b + noise

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# model
model = nn.Linear(1, 1)

# loss and optim
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_loss = []

# train
for epoch in range(1000):
    total_loss = 0
    iter_num = 0
    for x_train, y_train in dataloader:
        y_pred = model(x_train)
        loss_value = loss(y_pred, y_train)
        total_loss += loss_value.item()
        iter_num += 1
        # backward
        loss_value.backward()
        # update parameter
        optimizer.step()
        # clear the previous grad
        optimizer.zero_grad()

    # average loss
    train_loss.append(total_loss / iter_num)

print("w:", model.weight)
print("b:", model.bias)

# figure
plt.plot(train_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
plt.scatter(X, y)
y_pred = model.weight.item() * X + model.bias.item()
plt.plot(X, y_pred, color="red")
plt.show()