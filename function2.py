import numpy as np
import dask.array as da
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabaz_score
# from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def inspection_anomaly_detection(data, data_label, n_clusters):
    kf = KFold(n_splits=5, random_state=0)
    scores = np.zeros(5)
    for i, train_index, test_index in enumerate(kf.split(data, data_label)):
        train_data = data[train_index]
        k_means = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            random_state=0
        )
        k_means.fit(train_data)

        # <-------ここから閾値計算---------------------------------------->
        centroid = k_means.transform(train_data).min(axis=1)
        cluster_radius = np.zeros(k_means.cluster_centers_.shape[0])  # クラスタ半径
        for n_cluster in range(n_clusters):
            cluster_radius[n_cluster] = centroid[k_means.labels_ == n_cluster].max(
            )
        # <-------ここまで閾値計算---------------------------------------->
        test_data= data[test_index]
        anomaly = np.zeros(test_data.shape[0])
        # <-------ここから異常検知---------------------------------------->
        centroid_matrix = k_means.transform(test_data)
        predict_label = k_means.transform(test_data)
        predict = centroid_matrix < cluster_radius
        predict = predict.astype(np.int)
        for n_cluster in range(n_clusters):
            anomaly[predict_label==n_cluster] = predict[predict_label==n_cluster][:, n_cluster]
        # <-------ここまで異常検知---------------------------------------->
        test_label = data_label[test_index]
        # <-------ここから評価------------------------------------------->
        #    |異常 |正常
        # 異常| TP | FN
        # 正常| NP | TN
        # TP: 正しく異常を識別した
        # TN: 正しく正常を識別した
        # FP: 誤って異常と識別した
        # FN: 誤って正常と識別した
        scores[i] = f1_score(test_label, anomaly)
        # <-------ここまで評価------------------------------------------->
    return scores.mean()


def k_means(weight, n_clusters, data):

    print("クラスタ数(", n_clusters, ")個 -> ", "weight : ", weight,  end=" ")

    data = da.from_array(data, (1000, data.shape[1]))

    k_means = MiniBatchKMeans(
        n_clusters, max_iter=1000, random_state=0)
    k_means.fit((data * weight).compute())
    score = calinski_harabaz_score(data, k_means.labels_)
    # score = silhouette_score(data, k_means.labels_)
    # score = k_means.score(data)
    print("score : ", score)
    return -score
