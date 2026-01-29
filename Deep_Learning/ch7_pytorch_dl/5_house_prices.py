import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # 列转换器
from sklearn.pipeline import Pipeline # 管道操作
from sklearn.impute import SimpleImputer # 处理缺失值

from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt

import os

# 1. read data
def create_dataset():
    # 1.1 read data from source
    data = pd.read_csv('../data/house_prices.csv')

    # 1.2 delete unrelated features
    data.drop(['Id'], axis=1, inplace=True)

    # 1.3 划分特征和目标值
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    # 1.4 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 1.5 feature preprocess
    # 1.5.1 data
    numerical_features = X.select_dtypes(exclude=['object']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # 1.5.2
    numerical_transformer = Pipeline(steps=[
        ('fillNA', SimpleImputer(strategy='median')), # 用中位数填充数值类的NA
        ('scaler', StandardScaler()), # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('fillNA', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 1.5.3 构建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 1.5.4 create dataframe
    x_train = pd.DataFrame(preprocessor.fit_transform(x_train).toarray(), columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(preprocessor.transform(x_test).toarray(), columns=preprocessor.get_feature_names_out())

    # 1.5.5 create dataset
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())

    return train_dataset, test_dataset, x_train.shape[1] # num of features

# get data
train_dataset, test_dataset, n_features = create_dataset()
# print(test_dataset)
# print(n_features)

# NN model
model = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.BatchNorm1d(128), # batch normalization
    nn.ReLU(),
    nn.Dropout(0.2), # dropout
    nn.Linear(128, 1),
)

# define loss function
def log_rmse(y_pred, y_true):
    mse = nn.MSELoss()
    y_pred.squeeze_()
    # 进行范围限制
    y_pred = torch.clamp(y_pred, 1, float('inf'))
    return torch.sqrt(mse(torch.log(y_pred), torch.log(y_true)))

# train and test
def train_test(model, train_dataset, test_dataset, n_epochs, lr, batch_size, device):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    # init weight
    model.apply(init_weights)
    model.to(device)

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []

    # train
    for epoch in range(n_epochs):
        # train
        model.train()
        train_loss_total = 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch_count, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = log_rmse(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total += loss.item()

        # 这是每个batch的平均误差
        train_loss_avg = train_loss_total / len(train_loader)
        train_loss_list.append(train_loss_avg)

        # test
        model.eval()

        test_loss_total = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            for (X, y) in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = log_rmse(y_pred, y)

                test_loss_total += loss.item()

        test_loss_avg = test_loss_total / len(test_loader)
        test_loss_list.append(test_loss_avg)

        print(f"train loss: {train_loss_avg: .6f}, test loss: {test_loss_avg: .6f}")

    return train_loss_list, test_loss_list

print(os.getcwd())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr=0.1, n_epochs=100, batch_size=64, device=device)

plt.plot(train_loss_list, 'r-', label='train loss', linewidth=2)
plt.plot(test_loss_list, 'b--', label='test loss', linewidth=3)
plt.legend(loc='best')
plt.show()