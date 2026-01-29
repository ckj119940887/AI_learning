import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import light_year

from torch.utils.data import Dataset, DataLoader, TensorDataset

# data
data_train = pd.read_csv("../data/fashion-mnist_train.csv")
data_test = pd.read_csv("../data/fashion-mnist_test.csv")

# convert to tensor
# 转换成 N * 1 * 28 * 28
X_train = torch.tensor(data_train.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_train = torch.tensor(data_train.iloc[:, 0].values, dtype=torch.int64)

X_test = torch.tensor(data_test.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_test = torch.tensor(data_test.iloc[:, 0].values, dtype=torch.int64)

# plt.imshow(X_train[12345, 0, :, :], cmap='gray')
# plt.show()
# print(y_train[12345])

# define dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# model
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(in_features=16 * 5 * 5, out_features=120),
    nn.Sigmoid(),

    nn.Linear(in_features=120, out_features=84),
    nn.Sigmoid(),

    nn.Linear(in_features=84, out_features=10),
)

# check layers
# X = torch.rand((1, 1, 28, 28), dtype=torch.float)
# for layer in model:
#     X = layer(X)
#     print(layer.__class__.__name__, X.shape)

train_loss_list = []

# train model
def train_test(model, train_dataset, test_dataset, lr, n_epochs, batch_size, device):
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    model.apply(init_weights)
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct_count = 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_value = loss(y_pred, y)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            # batchsize可能除不尽，每个batch的数量可能不一样，所以这里直接将loss乘以batch的数量，数量多的权重占比多，数量少的权重占比少
            train_loss += loss_value.item() * X.shape[0]

            pred = y_pred.argmax(dim=1)
            train_correct_count += pred.eq(y).sum()
            print(f"\repoch:{epoch:0>2}[{'=' * (int((batch_idx + 1) / len(train_loader) * 50)):<50}]", end="")

        # compute avg loss，这里可以除以所有sample的数量的，因为前面乘以每个batch中sample的数量
        this_loss = train_loss / len(train_dataset)
        this_train_acc = train_correct_count / len(train_dataset)

        model.eval()
        test_correct_count = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                pred = y_pred.argmax(dim=1)
                test_correct_count += pred.eq(y).sum()

        this_test_acc = test_correct_count / len(test_dataset)

        print(f" loss:{this_loss:.6f},train_acc: {this_train_acc: .6f}, test_acc: {this_test_acc: .6f}")

train_test(model, train_dataset, test_dataset, lr=0.9, n_epochs=20, batch_size=256, device="cpu")
