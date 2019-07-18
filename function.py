import numpy as np
import dask.array as da
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import calinski_harabaz_score


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

def k_means_ch_score(weight, n_clusters, data, data_label):
    print("クラスタ数(", n_clusters, ")個 -> ", end=" ")

    data = da.from_array(data, ("auto", data.shape[1]))

    data = (data * weight).compute() # データと重みの要素積
    kf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
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
        scores[i] = calinski_harabaz_score(train_data, k_means.labels_)
        

def k_means(weight, n_clusters, data, data_label):
    print("クラスタ数(", n_clusters, ")個 -> ", end=" ")

    data = da.from_array(data, ("auto", data.shape[1]))

    data = (data * weight).compute() # データと重みの要素積
    kf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
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
    return -score
