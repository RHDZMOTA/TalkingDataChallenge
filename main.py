import logging
import json
import os

from optparse import OptionParser
from conf.settings import FilesConfig
from ml.evaluate import procedure as eval_procedure
from ml.search import procedure as random_procedure

RANDOM_SEARCH = "random-search"
MODEL_EVALUATION = "model-eval"
DATA = FilesConfig.Names.train_data


def check_options(opts):
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


def main(model, param, action):
    if action == RANDOM_SEARCH:
        random_search(model, param)
    if action == MODEL_EVALUATION:
        evaluate_model(model, param)


def evaluate_model(model, config_file):
    config = json.loads(open(config_file, "r").read())
    eval_procedure(DATA, model, config)


def random_search(model, n):
    print("%s %s" % (model, n))
    random_procedure(DATA, model, n)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model", type="string", help="ML Model")
    parser.add_option("--iter", type="string", help="Number of iterations (config option must not be provided.")
    parser.add_option("--config", type="string", help="Configuration file for model (iter option must not be provided).")
    # parser.add_options("--submit", action="store_true", help="Create submission files.")
    kwargs, _ = parser.parse_args(args=None, values=None)

    check_options(kwargs)
    action, param = determine_action(kwargs)
    main(getattr(kwargs, "model"), param, action)

