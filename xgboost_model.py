import gc
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
    "click_id": "uint32"
}

TRAIN_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "is_attributed"]
TEST_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "click_id"]
PREDICTION_COL = "is_attributed"


def main():
    data = get_data(split=0.7, frac=0.3)
    gc.collect()
    config = get_config()
    print(config)
    model = train_model(data["dtrain"], config)
    gc.collect()
    #dsubmit, results = get_submit_data()
    #results["is_attributed"] = model.predict(dsubmit, ntree_limit=model.best_ntree_limit)
    #results.to_csv("xgb_sub.csv", float_format="%0.8f", index=None)


def get_config():
    return {
        'eta': np.random.uniform(0, 1),
        'max_depth': np.random.choice(range(0, 11)),
        'min_child_weight': np.random.choice(range(1, 6)),
        'max_delta_step': np.random.choice(range(0, 11)),
        'subsample': np.random.uniform(0, 1),
        'colsample_bytree': np.random.uniform(0, 1),
        'colsample_bylevel': np.random.uniform(0, 1),
        'alpha': np.random.choice(range(0, 10)),
        'tree_method': 'hist', # gpu_hist
        'scale_pos_weight': np.random.choice(range(1, 11)),
        'grow_policy': np.random.choice(["depthwise", "lossguide"]),
        'max_leaves': int(np.random.uniform(0, 5000)),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 99,
        'silent': True
    }


def train_model(dtrain, config):
    watchlist = [(dtrain, 'train')]
    return xgb.train(config, dtrain, 30, watchlist, maximize=True, verbose_eval=1)


def get_data(split=0.7, frac=0.3):
    data = pd.read_csv("data/train.csv", usecols=TRAIN_COLS, dtype=DTYPES).sample(frac=frac)
    y = data[PREDICTION_COL]
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=1-split)
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


def get_submit_data():
    submit_data = pd.read_csv("data/test.csv", usecols=TEST_COLS, dtype=DTYPES)
    results = pd.DataFrame([])
    results["click_id"] = submit_data["click_id"]
    submit_data = create_features(submit_data)
    submit_data.drop(["click_id"], axis=1, inplace=True)
    return xgb.DMatrix(submit_data), results


def random_search():
    pass


if __name__ == "__main__":
    main()

