from ml.models import XGBoost


def evaluate_model(model_wrapper, config):
    trained_model = model_wrapper.train(config)
    return model_wrapper.get_test_auc(trained_model)


def procedure(data_path, model, config):
    if model == "xgboost":
        xgboost = XGBoost(data_path, split=0.5, frac=0.9)
        test_auc = evaluate_model(xgboost, config)
        print("(%s, %s)" % (model, str(test_auc)))



