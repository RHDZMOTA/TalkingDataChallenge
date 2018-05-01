import gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


TRAIN_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "is_attributed"]
TEST_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "click_id"]
PREDICTION_COL = "is_attributed"

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
    "click_id": "uint32"
}


def train_model(dtrain, config):
    watchlist = [(dtrain, 'train')]
    return xgb.train(config, dtrain, 30, watchlist, maximize=True, verbose_eval=0)


def get_data(data_path, split=0.7, frac=0.3):
    if "sample" in data_path:
        data = pd.read_csv(data_path, usecols=TRAIN_COLS, dtype=DTYPES).sample(frac=frac)
    else:
        data = read_large_file(data_path).sample(frac=frac)
    y = data[PREDICTION_COL]
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=1 - split)
    return {
        "dtrain": xgb.DMatrix(create_features(x_train), y_train),
        "dtest": xgb.DMatrix(create_features(x_test), y_test)
    }


def create_features(df):
    df = time_features(df)
    df = ip_counts(df)
    return df


def time_features(df):
    df["day_of_week"] = pd.to_datetime(df["click_time"]).dt.dayofweek
    return df.drop(["click_time"], axis=1)


def ip_counts(df):
    ip_count = df.groupby("ip")["channel"].count().reset_index()
    ip_count.columns = ["ip", "clicks_per_ip"]
    df = pd.merge(df, ip_count, on="ip", how="left", sort=False)
    return df.drop("ip", axis=1)


def read_large_file(data_path):
    #nchunks = 19
    #size = 184903890
    df = pd.DataFrame([])
    chunks = pd.read_csv(data_path, usecols=TRAIN_COLS, dtype=DTYPES, chunksize=10**7)
    for chunk in chunks:
        temp = chunk.sample(frac=0.05)
        df = df.append(temp)
        del temp
    gc.collect()
    return df


class XGBoost(object):

    def __init__(self, data_path, split=0.7, frac=0.3):
        self.data = get_data(data_path, split, frac)
        gc.collect()

    def train(self, config):
        return train_model(self.data["dtrain"], config)

    def get_train_auc(self, model):
        return roc_auc_score(self.data["dtrain"].get_label(), model.predict(self.data["dtrain"]))

    def get_test_auc(self, model):
        return roc_auc_score(self.data["dtest"].get_label(),  model.predict(self.data["dtest"]))
