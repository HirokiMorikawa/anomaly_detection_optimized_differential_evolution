import pprint
import numpy as np
from model import Model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    model = Model("creditcard.csv")
    feature = model.feature(scaling_proc=True)
    label = model.label()  # データのラベル

    print("データサイズ", feature.shape)
    print("正常", feature[label==0].shape)
    print("異常", feature[label==1].shape)


    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.4, shuffle=True, stratify=label)  # 学習データと検証データを分割する
    print("分割データ学習")
    print("データサイズ", X_train.shape)
    print("正常", X_train[y_train == 0].shape)  # 正常データ
    print("異常", X_train[y_train == 1].shape) # 異常データ
    print("分割データ検証")
    print("データサイズ", X_test.shape)
    print("正常", X_test[y_test == 0].shape) # 検証用データに含まれる正常データ
    print("異常", X_test[y_test == 1].shape)  # 検証用データに含まれるデータ    