# [Kaggle] Talking Data Challenge

This is my participation in the Kaggle Competition: [TalkingData AdTracking Fraud Detection Challenge
](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). 

**Requirements**
* Python 3.x
* Spark
* Kaggle CLI

## First-time setup

Previous requirements:
1. Install `unzip` with: `sudo apt install unzip`
1. Install the official [kaggle command line interface](https://github.com/Kaggle/kaggle-api).
1. Install spark and configure pyspark (python 3 as pyspark-driver) with this in your rc-file: 
    * `export PYSPARK_PYTHON=python3`
    * `export PYSPARK_DRIVER_PYTHON=python3`
1. Create an alias for spark submit in your rc-file:
    * `alias spark-submit=path/to/spark/bin/spark-submit`

To run this repo, follow these steps:

1. Clone into your machine.
1. Create the environmental variables: `cp conf/.env.example conf/.env`
1. Create a virtual environment:
    * `virtualenv --python=python3 venv`
    * `source venv/bin/activate`
1. Install requirements: `pip install -r requirements.txt`
1. Run the setup script: `bash setup.sh`

## Usage

This repo contains the code to random-search for a ml-model **or** 
run a particular configuration. 

### Random Search

To random search a model use the following command:
```bash
spark-submit main.py --model {name} --iter {n} {--submit}
```

Where:
```bash
    name : machine learning model name {xgboost, random-forest}
    n    : the number of random combination to search from e.g. 100
    --submit : add this flag to generate submit results (csv) for kaggle using the best model. 
```

Example:
```
spark-submit main.py --model xgboost --iter 25 --submit
```

### Model Evaluation

To evaluate a particular model use the following command:
```bash
spark-submit main.py --model {name} --config {file.json} {--submit}
```

Where:
```bash
    name      : machine learning model name {xgboost, random-forest}
    file.json : A json file containing the parameters of the model. 
    --submit  : add this flag to generate submit results (csv) for kaggle.
```

Example:
```
spark-submit main.py --model xgboost --config examples/xgboost_config.json --submit
```
### Model config 

To run a standalone model you need to provide a configuration file. 
You can see an example of this configuration files in the `examples/` dir.
#### XGBoost Config

This is an example configuration for the xgboost model:
```
{
  "eta": 0.660104014370388,
  "max_depth": 9, 
  "min_child_weight": 3, 
  "max_delta_step": 3, 
  "subsample": 0.16658770642331422, 
  "colsample_bytree": 0.628746219101687, 
  "colsample_bylevel": 0.42207502336694913, 
  "alpha": 8, 
  "tree_method": "hist", 
  "scale_pos_weight": 5, 
  "grow_policy": "lossguide", 
  "max_leaves": 1554, 
  "objective": "binary:logistic", 
  "eval_metric": "auc", 
  "random_state": 99, 
  "silent": true
}
```

#### Random Forest Config

[to be defined]