import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

# data
data_train = pd.read_csv("../data/fashion-mnist_train.csv")
data_test = pd.read_csv("../data/fashion-mnist_test.csv")

# convert to tensor
# 转换成 N * 1 * 28 * 28
X_train = torch.tensor(data_train.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_train = torch.tensor(data_train.iloc[:, 0].values, dtype=torch.int64)

X_test = torch.tensor(data_test.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_test = torch.tensor(data_test.iloc[:, 0].values, dtype=torch.int64)

plt.imshow(X_train[12345, 0, :, :], cmap='gray')
plt.show()
print(y_train[12345])