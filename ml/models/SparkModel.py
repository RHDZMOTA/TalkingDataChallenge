
from conf.settings import FilesConfig

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.sql.functions import hour, date_format
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession

LABEL_COL = "is_attributed"
FEATURES_COL = "features"

spark = SparkSession.builder.appName("talkdata-challenge").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

vector_assember = VectorAssembler(
    inputCols=["app", "device", "os", "channel", "week_day", "hour", "clicks_ip", "clicks_ip_app", "clicks_ip_app_os"],
    outputCol="features")


def get_random_forest_model(config, logger=None):
    if logger: logger.info("[function call] get_random_forest_model(config={})".format(str(config)))
    return RandomForestClassifier(labelCol=LABEL_COL, featuresCol=FEATURES_COL,
                                  numTrees=config["numTrees"], maxDepth=config["maxDepth"])


def get_gbt_model(config, logger=None):
    if logger: logger.info("[function call] get_gtb_model(config={})".format(str(config)))
    return GBTClassifier(labelCol=LABEL_COL, featuresCol=FEATURES_COL,
                         maxDepth=config["maxDepth"])


def train_model(model_str, train_dataset, config, logger=None):
    if model_str == "random-forest":
        return get_random_forest_model(config, logger).fit(train_dataset)
    if model_str == "gbt":
        return get_gbt_model(config, logger).fit(train_dataset)


def get_data(data_path, split=0.7, frac=0.3, logger=None):
    data, temp = spark.read.csv(data_path, header=True, inferSchema=True).randomSplit([frac, 1-frac])
    del temp
    data = create_features(data, logger)
    train, test = data.randomSplit([split, 1-split], seed=1)
    return {
        "train": vector_assember.transform(train).select([FEATURES_COL, LABEL_COL]),
        "test": vector_assember.transform(test).select([FEATURES_COL, LABEL_COL])
    }


def get_submit_data(logger=None):
    data = spark.read.csv(FilesConfig.Names.submit_data, header=True, inferSchema=True)
    data = create_features(data, logger)
    return vector_assember.transform(data).select([FEATURES_COL, "click_id"])


def create_features(df, logger=None):
    df = time_features(df, logger)
    df = perform_aggregations(df, logger)
    return df


def time_features(df, logger=None):
    if logger: logger.info("[function call] time_features(...)")
    df = df.withColumn("week_day", date_format("click_time", "u").cast(IntegerType()))
    df = df.withColumn("hour", hour("click_time"))
    return df


def perform_aggregations(df, logger=None):
    df = ip_counts(df, logger)
    df = ip_app_counts(df, logger)
    df = ip_app_os_counts(df, logger)
    return df


def ip_counts(df, logger=None):
    if logger: logger.info("[function call] ip_counts(...)")
    clicks_ip = df.groupBy("ip").count().withColumnRenamed("count", "clicks_ip")
    return df.join(clicks_ip, on=["ip"], how="left")


def ip_app_counts(df, logger=None):
    if logger: logger.info("[function call] ip_app_counts(...)")
    clicks_ip_app = df.groupBy(["ip", "app"]).count().withColumnRenamed("count", "clicks_ip_app")
    return df.join(clicks_ip_app, on=["ip", "app"], how="left")


def ip_app_os_counts(df, logger=None):
    if logger: logger.info("[function call] ip_app_os_counts(...)")
    clicks_ip_app_os = df.groupBy(["ip", "app", "os"]).count().withColumnRenamed("count", "clicks_ip_app_os")
    return df.join(clicks_ip_app_os, on=["ip", "app", "os"], how="left")


class SparkModel(object):

    def __init__(self, data_path, model_str, split=0.7, frac=0.3, logger=None):
        self.binary_evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
        self.model_str = model_str
        self.data = get_data(data_path, split, frac, logger)
        self.data["train"].cache()
        self.data["test"].cache()
        self.submit_tuple = None
        self.logger = logger

    def logger_wrapper(self, message):
        if self.logger: self.logger.info(message)

    def train(self, config):
        return train_model(self.model_str, self.data["train"], config)

    def get_train_auc(self, model):
        predictions = model.transform(self.data["train"])
        return self.binary_evaluator.evaluate(predictions)

    def get_test_auc(self, model):
        predictions = model.transform(self.data["test"])
        return self.binary_evaluator.evaluate(predictions)

    def create_submit_results(self, model, score=None):
        from util.timeformat import now
        submit_data = spark.read.csv(FilesConfig.Names.submit_data)
        submit_data = create_features(submit_data, self.logger)
        results = model.transform(submit_data)
        get_second_col = udf(lambda x: float(x.toArray()[1]))
        output_res = "folder-" + FilesConfig.Names.submit_output.format(
            date=now(only_date=False), model=self.model_str, score=score)
        results.withColumn("is_attributed", get_second_col(results["probability"])).select(
            ["click_id", "is_attributed"]).write.csv(output_res)
