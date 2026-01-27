import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # 列转换器
from sklearn.pipeline import Pipeline # 管道操作
from sklearn.impute import SimpleImputer # 处理缺失值

from torch.utils.data import Dataset, DataLoader, TensorDataset


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
print(test_dataset)
print(n_features)
