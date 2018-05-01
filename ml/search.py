import numpy as np

from ml.evaluate import evaluate_model
from ml.models import XGBoost


def get_config(model):
    if model == "xgboost":
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


def procedure(data_path, model, n):
    if model == "xgboost":
        xgboost = XGBoost(data_path, split=0.7, frac=0.8)
        configs = []
        for i in range(int(n) + 1):
            config = get_config(model)
            test_auc = evaluate_model(xgboost, config)
            print(test_auc)
            config["auc"] = test_auc
            configs.append(config)
