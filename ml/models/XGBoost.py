import gc
import pandas as pd
import xgboost as xgb
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from conf.settings import FilesConfig

warnings.filterwarnings("ignore")

TRAIN_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "is_attributed"]
TEST_COLS = ["ip", "app", "device", "os", "channel" , "click_time", "click_id"]
PREDICTION_COL = "is_attributed"
TRAIN_LARGE_FILE_PERCENTAGE = 0.15

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
    "click_id": "uint32"
}


def train_model(dtrain, config, logger=None):
    if logger: logger.info("[function call] train_model(...)")
    watchlist = [(dtrain, 'train')]
    return xgb.train(config, dtrain, 30, watchlist, maximize=True, verbose_eval=0)


def get_data(data_path, split=0.7, frac=0.3, logger=None):
    if logger: logger.info("[function call] get_data(data_path=%s, split=%s, frac=%s)" %
                           (data_path, str(split), str(frac)))
    if "sample" in data_path:
        data = pd.read_csv(data_path, usecols=TRAIN_COLS, dtype=DTYPES).sample(frac=frac)
    else:
        data = read_large_file(data_path, logger).sample(frac=frac)
    y = data[PREDICTION_COL]
    data.drop(PREDICTION_COL, axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=1 - split)
    return {
        "dtrain": xgb.DMatrix(create_features(x_train, logger), y_train),
        "dtest": xgb.DMatrix(create_features(x_test, logger), y_test)
    }


def get_submit_data(logger=None):
    if logger: logger.info("[function call] get_submit_data()")
    submit_data = pd.read_csv(FilesConfig.Names.submit_data, usecols=TEST_COLS, dtype=DTYPES)
    results = submit_data[["click_id"]]
    return xgb.DMatrix(create_features(submit_data.drop("click_id", axis=1), logger)), results


def create_features(df, logger=None):
    if logger: logger.info("[function call] create_features(...)")
    df = time_features(df, logger)
    df = ip_counts(df, logger)
    return df


def time_features(df, logger=None):
    if logger: logger.info("[function call] time_features(...)")
    df["day_of_week"] = pd.to_datetime(df["click_time"]).dt.dayofweek
    return df.drop(["click_time"], axis=1)


def ip_counts(df, logger=None):
    if logger: logger.info("[function call] ip_counts(...)")
    ip_count = df.groupby("ip")["channel"].count().reset_index()
    ip_count.columns = ["ip", "clicks_per_ip"]
    df = pd.merge(df, ip_count, on="ip", how="left", sort=False)
    return df.drop("ip", axis=1)


def read_large_file(data_path, logger=None):
    if logger: logger.info("[function call] read_large_file(data_path=%s) with TRAIN_LARGE_FILE_PERCENTAGE=%s" %
                           (data_path, str(TRAIN_LARGE_FILE_PERCENTAGE)))
    #nchunks = 19
    #size = 184903890
    df = pd.DataFrame([])
    chunks = pd.read_csv(data_path, usecols=TRAIN_COLS, dtype=DTYPES, chunksize=10**7)
    for chunk in chunks:
        temp = chunk.sample(frac=TRAIN_LARGE_FILE_PERCENTAGE)
        df = df.append(temp)
        del temp
    gc.collect()
    return df


class XGBoost(object):

    def __init__(self, data_path, split=0.7, frac=0.3, logger=None):
        if logger: logger.info("[class creation] XGBoost(data_path=%s, split=%s, frac=%s)" %
                               (data_path, str(split), str(frac)))
        self.data = get_data(data_path, split, frac, logger)
        self.submit_tuple = None
        self.logger = logger
        gc.collect()

    def logger_wrapper(self, message):
        if self.logger: self.logger.info(message)

    def train(self, config):
        self.logger_wrapper("[method call] XGBoost(...).train(config=%s)" % str(config))
        return train_model(self.data["dtrain"], config, self.logger)

    def get_train_auc(self, model):
        self.logger_wrapper("[method call] XGBoost(...).get_train_auc(model=...)")
        return roc_auc_score(self.data["dtrain"].get_label(), model.predict(self.data["dtrain"]))

    def get_test_auc(self, model):
        self.logger_wrapper("[method call] XGBoost(...).get_test_auc(model=...)")
        return roc_auc_score(self.data["dtest"].get_label(),  model.predict(self.data["dtest"]))

    def create_submit_results(self, model, score=None):
        from util.timeformat import now
        self.logger_wrapper("[method call] XGBoost(...).create_submit_results(...)")
        if not self.submit_tuple:
            sdf, results = get_submit_data(self.logger)
            self.submit_tuple = (sdf, results)
        else:
            sdf, results = self.submit_tuple
        results["is_attributed"] = model.predict(sdf, ntree_limit=model.best_ntree_limit)
        score = ("%0.4f" % score) if score else "none"
        results.to_csv(
            FilesConfig.Names.submit_output.format(date=now(only_date=False), model="xgboost", score=score),
            float_format="%0.8f", index=None)
