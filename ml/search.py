import gc
import numpy as np
import pandas as pd

from util.timeformat import now
from conf.settings import FilesConfig
from ml.evaluate import evaluate_model
from ml.models import XGBoost


def get_config(model):
    if model == "xgboost":
        return {
            'eta': np.random.uniform(0, 1),
            'max_depth': np.random.choice(range(1, 11)),
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


def procedure(data_path, model, n, submit, logger):
    if logger: logger.info("[function call] ml.models.search.procedure(data_path=%s, model=%s, n=%s, submit=%s)" %
                           (data_path, model, str(n), str(submit)))
    if model == "xgboost":
        xgboost = XGBoost(data_path, split=0.7, frac=0.8, logger=logger)
        rand_search(n, model, xgboost, submit, logger)


def rand_search(n, model, model_wrapper, submit, logger):
    configs = []
    best_model = (-1, None)
    for i in range(int(n) + 1):
        config = get_config(model)
        trained_model, test_auc = evaluate_model(model_wrapper, config)
        if logger: logger.info("[procedure] [random search: random] (model=%s, auc=%0.4f)" % (model, test_auc))
        config["auc"] = test_auc
        configs.append(config)
        if test_auc > best_model[0]:
            best_model = (test_auc, trained_model)
        gc.collect()
    score = best_model[0]
    score = ("%0.4f" % score) if score else "none"
    if logger: logger.info("[results] [random search: best model] (model=%s, auc=%s)" % (model, score))
    pd.DataFrame(configs).to_csv(
        FilesConfig.Names.results.format(date=now(only_date=False), model=model, score=score),
        index=None)
    if submit:
        test_auc, trained_model = best_model
        model_wrapper.create_submit_results(trained_model, test_auc)
