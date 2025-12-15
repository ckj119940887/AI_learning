import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    data = pd.read_csv("../data/train.csv")
    x = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_train, x_test, y_train.values, y_test.values

if __name__ == "__main__":
    load_data()