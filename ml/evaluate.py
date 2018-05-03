from ml.models import XGBoost, SparkModel


def evaluate_model(model_wrapper, config):
    trained_model = model_wrapper.train(config)
    return trained_model, model_wrapper.get_test_auc(trained_model)


def procedure(data_path, model, config, submit, logger):
    if logger: logger.info("[function call] ml.evaluate.procedure(data_path=%s, model=%s, config=...)" %
                           (data_path, model))
    if model == "xgboost":
        xgboost = XGBoost(data_path, split=0.5, frac=0.9, logger=logger)
        trained_model, test_auc = evaluate_model(xgboost, config)
        if logger: logger.info("[results] [evaluate] (model=%s, auc=%0.4f)" % (model, test_auc))
        if submit:
            xgboost.create_submit_results(trained_model, test_auc)
    else:
        spark_model = SparkModel(data_path, model, split=0.5, logger=logger)
        trained_model, test_auc = evaluate_model(spark_model, config)
        if logger: logger.info("[results] [evaluate] (model=%s, auc=%0.4f)" % (model, test_auc))
        if submit:
            spark_model.create_submit_results(trained_model, test_auc)

