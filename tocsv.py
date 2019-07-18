import pandas as pd

if __name__=="__main__":
    df = pd.read_json("異常検知0混同行列_重み.txt")
    print(df)