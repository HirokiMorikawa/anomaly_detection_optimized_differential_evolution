import numpy as np
import dask.array as da
import pickle
import pprint

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from model import Model


"""
異常検知で評価の良いクラスタ数を探索
"""

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
    # <-------ここまで閾値計算---------------------------------------->
    anomaly = np.zeros(test_data.shape[0])
    # <-------ここから異常検知---------------------------------------->
    centroid_matrix = km.transform(test_data)
    predict_label = km.predict(test_data)
    predict = centroid_matrix > cluster_radius
    predict = predict.astype(np.int)
    for n_cluster in range(n_clusters):
        anomaly[predict_label == n_cluster] = predict[predict_label ==
                                                      n_cluster][:, n_cluster]
    # <-------ここまで異常検知---------------------------------------->

    # <-------ここから評価------------------------------------------->
    #    |異常 |正常
    # 異常| TP | FN
    # 正常| NP | TN
    # TP: 正しく異常を識別した
    # TN: 正しく正常を識別した
    # FP: 誤って異常と識別した
    # FN: 誤って正常と識別した
    score = f1_score(test_label, anomaly)
    # <-------ここまで評価------------------------------------------->
    return score


def k_means(n_clusters, data, data_label):
    print("クラスタ数(", n_clusters, ")個 -> ", end=" ")

    kf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True) #
    scores = np.zeros(3)
    for i, index in enumerate(kf.split(data, data_label)):
        train_index, test_index = index
        train_data = data[train_index][data_label[train_index] == 0]
        k_means = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            random_state=0
        )
        k_means.fit(train_data)
        scores[i] = anomaly_detection(
            k_means, train_data, data[test_index], data_label[test_index])
    score = np.mean(scores)
    print("score : ", score)
    return score

if __name__ == "__main__":
    model = Model("creditcard.csv")
    feature = model.feature(scaling_proc=True)
    label = model.label()  # データのラベル
    scores = {}
    for c in range(2, 101):
        scores[c] = k_means(c, feature, label)
    pprint.pprint(scores)


    
    
