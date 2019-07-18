import pandas as pd
import pickle
from scipy.optimize import differential_evolution as de

def load_data(file_name):
    """
    引数にファイル名を与えると，その名前のファイルを読み込んでファイルの内容を
    pandasのDataFrameにして返す関数
    """
    df = pd.read_csv(file_name)
    return df

def load_model(file_name) -> de:
    """
    引数にファイル名を与えると，その名前のファイルを読み込んでpickle漬けされた差分進化の計算結果を
    返す関数
    """
    # pickle.load(file_name)
    with open(file_name, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    data = load_data("creditcard.csv")
    print(data)
