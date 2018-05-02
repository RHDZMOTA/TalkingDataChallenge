import logging
import json
import os

from optparse import OptionParser
from conf.settings import FilesConfig, LogConf
from ml.evaluate import procedure as eval_procedure
from ml.search import procedure as random_procedure

RANDOM_SEARCH = "random-search"
MODEL_EVALUATION = "model-eval"
DATA = FilesConfig.Names.train_data


def check_options(opts):
    logger.info("[function call] check_options(opts=%s)" % str(opts))
    model_opt = getattr(opts, "model")
    iter_opt = getattr(opts, "iter")
    config_opt = getattr(opts, "config")
    valid_models = ["xgboost", "random-forest"]

    if model_opt is None:
        raise ValueError("A model must be provided.")

    if model_opt not in valid_models:
        raise ValueError("Invalid model: %s not in %s" % (model_opt, str(valid_models)))

    if iter_opt:
        int(iter_opt)

    if config_opt:
        if not os.path.exists(config_opt):
            raise ValueError("File %s does not exists." % config_opt)

    if sum([isinstance(e, str) for e in [iter_opt, config_opt]]) != 1:
        raise ValueError("Exactly one flag must be used: either --iter or --config")


def determine_action(opts):
    config_opt = getattr(opts, "config")
    if not config_opt:
        return RANDOM_SEARCH, getattr(opts, "iter")
    return MODEL_EVALUATION, config_opt


def main(model, param, action, submit):
    if action == RANDOM_SEARCH:
        random_search(model, param, submit)
    if action == MODEL_EVALUATION:
        evaluate_model(model, param, submit)


def evaluate_model(model, config_file, submit):
    logger.info("[function call] evaluate_model(model=%s, config_file=%s, submit=%s)" %
                (model, config_file, str(submit)))
    config = json.loads(open(config_file, "r").read())
    eval_procedure(DATA, model, config, submit, logger)


def random_search(model, n, submit):
    logger.info("[function call] random_search(model=%s, n=%s, submit=%s)" % (model, str(n), str(submit)))
    random_procedure(DATA, model, n, submit, logger)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--submit", action="store_true", help="Create submission files.")
    parser.add_option("--model", type="string", help="ML Model")
    parser.add_option("--iter", type="string", help="Number of iterations (config option must not be provided.")
    parser.add_option("--config", type="string", help="Configuration file for model (iter option must not be provided).")
    kwargs, _ = parser.parse_args(args=None, values=None)

    logger = LogConf.create(logging)
    check_options(kwargs)
    action, param = determine_action(kwargs)
    main(getattr(kwargs, "model"), param, action, getattr(kwargs, "submit"))

