import os
import numpy as np
import pickle
import pprint

from sys import argv
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution as de
from sklearn.model_selection import train_test_split
from model import Model

import function


class Optimizer:
    def __init__(self, func, bounds, vec_size=1, pop_size=15, cr=0.9, f=0.5, max_iter=None, callback=None):
        self.__bounds = bounds
        self.__func = func
        self.__pop_size = pop_size
        self.__cr = cr
        self.__f = f
        self.__callback = None
        self.__max_iter = None
        if max_iter is not None and type(max_iter) == int:
            self.__max_iter = max_iter
        else:
            self.__max_iter = 100

    def run(self, seed=0, func_params=None):
        return de(
            func=self.__func,
            # strategy="randtobest1bin",
            args=func_params,
            maxiter=self.__max_iter,
            bounds=self.__bounds,
            popsize=self.__pop_size,
            mutation=self.__f,
            recombination=self.__cr,
            callback=self.__callback,
            polish=False,
            updating='deferred',
            workers=max(1, cpu_count() - 1),
            seed=seed,
            disp=True
        )


if __name__ == "__main__":
    if len(argv) == 1:
        directory = "./de_result"
    else:
        directory = argv[1]

    model = Model("creditcard.csv")
    feature = model.feature(scaling_proc=True)
    label = model.label()  # データのラベル

    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.4, shuffle=True, stratify=label)  # 学習データと検証データを分割する

    # X_train_regular = X_train[y_train == 0]  # 正常データ
    # X_train_anomaly = X_train[y_train == 1] # 異常データ
    # X_test_regular = X_test[y_test == 0]  # 検証用データに含まれる正常データ
    # X_test_anomaly = X_test[y_test == 1]  # 検証用データに含まれるデータ
    # print(X_train.shape)
    # print(X_train_regular.shape)
    # print(X_train_anomaly.shape)
    # print(X_test.shape)
    # print(X_test_regular.shape)
    # print(X_test_anomaly.shape)

    # bounds = [(0, 1) for i in range(feature.shape[1])]
    # opt = Optimizer(function.k_means, [(1, 20), (0, 1)], 2, cr=0.5, f=0.1, pop_size=20, max_iter=1000)
    # opt = Optimizer(function.k_means, [(1, 20), (1, 5000), (0, 1)], 3, cr=0.5, f=0.1, pop_size=20, max_iter=1000)
    # opt = Optimizer(function.k_means, [(2, 200), (1, 20), (0, 1)], 3, cr=0.9, f=0.1, pop_size=20, max_iter=1000)

    # directory = "./de_result"
    if not os.path.exists(directory):
        os.mkdir(directory)
    opt = Optimizer(function.k_means, [(0, 1) for i in range(
        feature.shape[1])], feature.shape[1], cr=0.5, f=0.1, pop_size=10, max_iter=100)
    for seed in range(0, 21):
        result = opt.run(seed=seed, func_params=(18, X_train, y_train))
        pprint.pprint(result)

        with open(directory + "/" + "weight_" + str(seed) + ".pickle", "wb") as f:
            pickle.dump(result, f, 4)
