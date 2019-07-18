from loader import load_data
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

import pandas as pd


class Model:
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__data = load_data(file_name)

    def reload(self):
        self.__data = load_data(self.__file_name)

    def feature(self, scaling_proc=False):
        _feature = self.__data.iloc[:, 1:-1]

        if not scaling_proc:
            return _feature.values
        
        scaler = MinMaxScaler()
        _feature = scaler.fit_transform(_feature)
        return _feature
        
    
    def label(self):
        return self.__data["Class"].values
    
if __name__ == "__main__":
    model = Model("creditcard.csv")
    feature = model.feature(scaling_proc=True)
    label = model.label()
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, shuffle=False)
    
    X_train_regular = X_train[y_train==0] # 正常データ
    X_test_regular = X_test[y_test==0]
    X_test_anomaly = X_test[y_test==1]
    print("普通の")
    k_means = KMeans(n_clusters=8, max_iter=1000,verbose=1, random_state=0)
    pprint(k_means.fit(X_train))
    print("ミニバッチ")
    k_means = MiniBatchKMeans(n_clusters=8, max_iter=1000, verbose=1, random_state=0)
    pprint(k_means.fit(X_train))
    
