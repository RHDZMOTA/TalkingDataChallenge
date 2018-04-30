import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression

from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, date_format
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("talkdata-challenge").getOrCreate()
spark.conf.set("spark.driver.memory", "12g")
spark.conf.set("spark.executor.memory", "12g")

print(spark.conf.get("spark.driver.memory"))

# READ DATASET
print("======================== Read dataset ========================")
dataset = spark.read.csv("data/train.csv", header=True, inferSchema=True)
#dataset = dataset.withColumn("week_day", date_format("click_time", "u").cast(IntegerType()))
#dataset = dataset.withColumn("hour", hour("click_time"))

#dataset.cache()
#dataset.show()


# CREATE FEATURES
print("======================== Vector Assembler ========================")

def transformDataset(df, vectorAssembler, outputCol=None):
    df = df.withColumn("week_day", date_format("click_time", "u").cast(IntegerType()))
    df = df.withColumn("hour", hour("click_time"))
    clicks_ip = df.groupBy("ip").count()
    df = df.join(clicks_ip, on=["ip"], how="left")
    if outputCol:
        return vectorAssembler.transform(df).select(["features", outputCol])
    return vectorAssember.transform(df)

vectorAssember = VectorAssembler(
            inputCols=["app", "device", "os", "channel", "week_day", "hour", "count"],
            outputCol="features")


feature_dataset = transformDataset(dataset, vectorAssember, "is_attributed")

#feature_dataset.cache()
#feature_dataset.show()


# CREATE TEST / TRAIN DATASETS

print("======================== Split Data ========================")
train, test = feature_dataset.randomSplit([0.6, 0.4], seed=1)

print("======================== Cache Test/Train ========================")
train.cache()
test.cache()


# MODELS

def randomForestModelFunction(numTrees, maxDepth): 
    randomForestModel = RandomForestClassifier(
        labelCol="is_attributed",
        featuresCol="features",
        numTrees=numTrees,
        maxDepth=maxDepth).fit(train)
            
    randomForestTrainPredictions = randomForestModel.transform(train)
    randomForestTestPredictions = randomForestModel.transform(test)
    return randomForestTrainPredictions, randomForestTestPredictions, randomForestModel


def gradientBoostingModelFunction(maxIter):
    gradientBoostingModel = GBTClassifier(
            labelCol="is_attributed",
            featuresCol="features",
            maxIter=maxIter).fit(train)
            
    gbTrainPredictions = gradientBoostingModel.transform(train)
    gbTestPredictions = gradientBoostingModel.transform(test)
    return gbTrainPredictions, gbTestPredictions


# EVALUATOR

print("======================== Instance Binary Evaluator ========================")
binaryEvaluator = BinaryClassificationEvaluator(labelCol="is_attributed", metricName="areaUnderROC")

# RANDOM FOREST

def createRandomForestConfig():
        return {
            "numTrees": int(np.random.uniform(10, 140)),
            "maxDepth": int(np.random.uniform(2, 10))
        }

def randomForestEval(returnModel=False):
    config = createRandomForestConfig()
    trainPred, testPred, model = randomForestModelFunction(config["numTrees"], config["maxDepth"])
    config["train"] = binaryEvaluator.evaluate(trainPred)
    config["test"] = binaryEvaluator.evaluate(testPred)
    if returnModel:
        return config, model
    return config


def randomForestResults(nSearch, plot=True):
    configs = []
    for i in range(nSearch + 1):
        configs.append(randomForestEval())
        last = configs[-1]
        print("%d)\t(%d, %d)\t=>\t%f" % (i, last["numTrees"], last["maxDepth"], last["test"]))
        pd.DataFrame(configs).to_csv("temp2.csv", index=None)

    randomForestResultDf = pd.DataFrame(configs)

    def scatter(x, y, **kwargs):
        kwargs.pop("color")
        plt.scatter(x, y, **kwargs)

    if plot:
        fg = sns.FacetGrid(data=randomForestResultDf, aspect=2)
        fg.map(scatter, "maxDepth", "numTrees", c=randomForestResultDf.test, s=50, cmap = 'seismic')

        plt.colorbar()
        plt.title("[Random Search] Random Forest Classifier")
        plt.ylabel("Number of Trees")
        plt.xlabel("Max Depth")
        plt.show()
        plt.close()


# GRADIENT BOOSTING

def createGradientBoostingConfig():
    return {
        "maxIter": int(np.random.uniform(2, 100)) 
    }

def gradientBoostingEval():
    config = createGradientBoostingConfig()
    trainPred, testPred = gradientBoostingModelFunction(config["maxIter"])
    config["train"] = binaryEvaluator.evaluate(trainPred)
    config["test"] = binaryEvaluator.evaluate(testPred)
    return config


def gradientBoostingResults(nSearch, plot=True):
    configs = []
    for i in range(nSearch + 1):
        configs.append(gradientBoostingEval())
        last = configs[-1] 
        print("%d)\t(%d)\t=>\t%f" % (i, last["maxIter"], last["test"]))
        pd.DataFrame(configs).to_csv("gb_results.csv", index=None)

    gbResults = pd.DataFrame(configs)

    if plot:
        fg = sns.FacetGrid(data=gbResults, aspect=2)
        fg.map(plt.scatter, "maxIter", "test")

        plt.title("[Random Search] Gradient Boosting Classifier")
        plt.ylabel("ROC score")
        plt.xlabel("Max Iter")
        plt.show()
        plt.close()


# Ensemble method

def bestModel():
    randomForestConfig = {"maxDepth": 9, "numTrees": 62}
    gradientBoostingConfig = {"maxIter": 70}

    # Train and evaluate model
    print("======================== Training Model ========================")
    rfTrainPred, rfTestPred, model = randomForestModelFunction(randomForestConfig["numTrees"], randomForestConfig["maxDepth"])

    print("======================== Evaluator Eval ========================")
    rfTrainEval = binaryEvaluator.evaluate(rfTrainPred)
    rfTestEval = binaryEvaluator.evaluate(rfTestPred)

    print("======================== Results ========================")
    print(rfTestEval)
    print("\nRandom forest results: {train: %f , test: %f}\n" % (rfTrainEval, rfTestEval))

    return model

print("======================== Function Call: Best Model ========================")
model = bestModel()


print("======================== Read submission data ========================")
submit_data = spark.read.csv("data/test.csv", header=True, inferSchema=True).cache()

print("======================== Generate submition features ========================")
submit_features = transformDataset(submit_data, vectorAssember)

print("======================== Transform sumbission data ========================")
submit_results = model.transform(submit_features)
submit_results.show()
print(submit_results.printSchema())
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
getSecond = udf(lambda x: float(x.toArray()[1]))
#submit_results.withColumn("is_attributed", getSecond(submit_results["probability"])).show() 
print("======================== Write Data ========================")
submit_results.withColumn("is_attributed", getSecond(submit_results["probability"])).select(["click_id", "is_attributed"]).write.csv("test_complete_spark")#("submission_complete.csv")
