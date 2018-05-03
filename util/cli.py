import os

from optparse import OptionParser

RANDOM_SEARCH = "random-search"
MODEL_EVALUATION = "model-eval"


class CLI(object):

    def __init__(self):
        parser = OptionParser()
        parser.add_option("--submit", action="store_true", help="Create submission files.")
        parser.add_option("--sample", action="store_true", help="Use sample train dataset.")
        parser.add_option("--logging", action="store_true", help="Log execution.")
        parser.add_option("--model", type="string", help="ML Model")
        parser.add_option("--iter", type="string", help="Number of iterations (config option must not be provided.")
        parser.add_option("--config", type="string", help="Configuration file for model (iter option must not be provided).")
        self.opts, _ = parser.parse_args(args=None, values=None)
        self.check_options()

    def check_options(self):
        model_opt = getattr(self.opts, "model")
        iter_opt = getattr(self.opts, "iter")
        config_opt = getattr(self.opts, "config")
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

    def determine_action(self):
        config_opt = getattr(self.opts, "config")
        if not config_opt:
            return RANDOM_SEARCH, getattr(self.opts, "iter")
        return MODEL_EVALUATION, config_opt

    def get(self, key):
        return getattr(self.opts, key)
