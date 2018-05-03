import logging
import json

from conf.settings import FilesConfig, LogConf
from ml.evaluate import procedure as eval_procedure
from ml.search import procedure as random_procedure
from util.cli import CLI

RANDOM_SEARCH = "random-search"
MODEL_EVALUATION = "model-eval"


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
    logger = LogConf.create(logging)
    cli = CLI(logger)
    action, param = cli.determine_action()
    DATA = FilesConfig.Names.train_sample_data if cli.get("sample") else FilesConfig.Names.train_data
    main(cli.get("model"), param, action, cli.get("submit"))

