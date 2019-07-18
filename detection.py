
import numpy as np
import dask.array as da
import json
from model import Model
from loader import load_model
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metric 
from pprint import pprint
import pandas as pd

def dic_trans(d: dict):
    return {key: d[key].tolist() for key in d.keys()}

class Result:
    def __init__(self, size=0):
        self._i = 0
        self.size = size
        self.f1_score = None
        self.confusion_matrix = None
        if size > 0:
            self.f1_scores = np.zeros(size)
            self.confusion_matrixs = {i: None for i in range(size)}

    
    def mean_score(self):
        return self.f1_score.mean()
    
    def add_result(self, f1_score, confusion_matrix):
        self.f1_score = f1_score
        self.confusion_matrix = confusion_matrix
        if self.size > 0:
            self.f1_scores[self._i] = f1_score
            self.confusion_matrixs[self._i] = confusion_matrix
            self._i += 1
        


def anomaly_detection(km, train_data, test_data, test_label):
    """
    テストデータの異常検知を行う関数
    """
    n_clusters = km.n_clusters

    # <-------ここから閾値計算---------------------------------------->
    centroid = da.from_array(km.transform(train_data), chunks=("auto", train_data.shape[1])) # データ処理の並列化を行う optional
    centroid = centroid.min(axis=1)
    # centroid = km.transform(train_data).min(axis=1)  # クラスタ距離で最短となるものを求める
    cluster_radius = np.zeros(km.cluster_centers_.shape[0])  # クラスタ半径
    for n_cluster in range(n_clusters):  # 各クラスタの半径を求める
        cluster_radius[n_cluster] = (centroid[km.labels_ == n_cluster].max()).compute()
        # cluster_radius[n_cluster] = centroid[km.labels_ == n_cluster].max(
        # )
    # print(cluster_radius)
    # <-------ここまで閾値計算---------------------------------------->
    anomaly = np.zeros(test_data.shape[0])
    # <-------ここから異常検知---------------------------------------->
    centroid_matrix = km.transform(test_data)
    predict_label = km.predict(test_data)
    predict = centroid_matrix > cluster_radius
    # print(predict)
    predict = predict.astype(np.int)
    for n_cluster in range(n_clusters):
        anomaly[predict_label == n_cluster] = predict[predict_label ==
                                                      n_cluster][:, n_cluster]
    # print(anomaly)
    # <-------ここまで異常検知---------------------------------------->

    # <-------ここから評価------------------------------------------->
    # f値と混合行列
    #    |異常 |正常
    # 異常| TP | FN
    # 正常| FP | TN
    # TP: 正しく異常を識別した
    # TN: 正しく正常を識別した
    # FP: 誤って異常と識別した
    # FN: 誤って正常と識別した
    score = metric.f1_score(test_label, anomaly)
    c_matrix = metric.confusion_matrix(test_label, anomaly)
    # <-------ここまで評価------------------------------------------->
    ret = Result()
    ret.add_result(score, c_matrix)
    return ret

def detection(n_clusters, model, weight=None, seed=-1):
    print("{}, {} ->".format(n_clusters, seed), end=" ")

    feature = model.feature(scaling_proc=True)
    label = model.label()  # データのラベル

    if weight is not None:
        # print("weight -> {}".format(weight), end="")
        feature = da.from_array(feature, chunks=("auto", feature.shape[1]))
        feature = (feature * weight).compute()

    if seed < 0:
        X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.4, shuffle=True, stratify=label)  # 学習データと検証データを分割する
    else:
        X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.4, shuffle=True, stratify=label, random_state=seed)  # 学習データと検証データを分割する
    
    kf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    result_train = Result(3)
    result_test = Result(3)
    for i, index in enumerate(kf.split(X_train, y_train)):
        train_index, test_index = index
        train_data = X_train[train_index][y_train[train_index] == 0]
        k_means = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter = 1000,
            random_state=0
        )
        k_means.fit(train_data)
        _train = anomaly_detection(k_means, train_data, X_train[test_index], y_train[test_index])
        _test = anomaly_detection(k_means, train_data, X_test, y_test)

        result_train.add_result(_train.f1_score, _train.confusion_matrix)
        result_test.add_result(_test.f1_score, _test.confusion_matrix)
    
    # k_means = MiniBatchKMeans(
    #         n_clusters=n_clusters,
    #         max_iter = 1000,
    #         random_state=0
    #     )
    # k_means.fit(X_train)

    # return result_train.mean_score(), result_test.mean_score()
    return [result_train, result_test]
    # return anomaly_detection(k_means, X_train, X_test, y_test)


if __name__ == "__main__":
    # print(load_model("./detection_model_nagato/weight_0.pickle"))
    model = Model("creditcard.csv")


    for div_seed in range(5):
        _, no_weight = detection(18, model, seed=div_seed)
        print(no_weight.mean_score())
        np.savetxt("異常検知結果{}.txt".format(div_seed), [no_weight.mean_score()], fmt='%.5f')
        with open("異常検知結果{}混合行列.txt".format(div_seed), "w") as f:
            json.dump(dic_trans(no_weight.confusion_matrixs), f, indent=4)
        # train, test = no_weight
        # pprint(test.mean_score())
        # pprint(test.confusion_matrixs)
        score = np.zeros(20)
        m = {i : None for i in range(20)}
        for seed in range(0, 20):
            print("de: {}, ".format(seed) , end="")
            de = load_model("./detection_model_nagato/weight_{}.pickle".format(seed))
            _, weighted = detection(18, model, weight=de.x, seed=div_seed)
            batch_score = weighted.mean_score()
            print(batch_score)
            score[seed] = batch_score
            m[seed] = dic_trans(weighted.confusion_matrixs)
        np.savetxt("異常検知結果{}_重み.txt".format(div_seed), [score.mean()], fmt='%.5f')
        with open("異常検知結果{}混合行列_重み.txt".format(div_seed), "w") as f:
            json.dump(m, f, indent=4)
        
        m2 = np.zeros((20, 4),int)

        index = np.arange(20)
        columns = ["TN", "FP", "FN", "P"]

        for seed in range(0, 20):
            ss = np.empty((0,4), int)
            for i in range(3):
                f = np.array(m[seed][i], int).flatten()
                # print(f.shape)
                ss = np.append(ss, [f], axis=0)
            m2[seed] = ss.mean(axis=0)
        
        df = pd.DataFrame(data=m2, index=index, columns=columns)
        df.to_csv("異常検知結果(仮){}混合行列_重み.txt".format(div_seed))