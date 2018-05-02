# [Kaggle] Talking Data Challenge

This is my participation in the Kaggle Competition: [TalkingData AdTracking Fraud Detection Challenge
](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). 

**Requirements**
* Python 3.x
* Spark
* Kaggle CLI

## First-time setup

To run this repo, follow these steps:

1. Clone into your machine.
2. Create the environmental variables: `cp conf/.env.example conf/.env`
3. Create a virtual environment:
    * `virtualenv --python=python3 venv`
    * `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Run the setup script: `bash setup.sh`

## Usage

This repo contains the code to random-search for a ml-model **or** 
run a particular configuration. 

### Random Search

To random search a model use the following command:
```bash
python main.py --model {name} --iter {n}
```

Where:
```bash
    name : machine learning model name {xgboost, random-forest}
    n    : the number of random combination to search from e.g. 100
```

### Model Evaluation

To evaluate a particular model use the following command:
```bash
python main.py --model {name} --config {file.json}
```

Where:
```bash
    name      : machine learning model name {xgboost, random-forest}
    file.json : A json file containing the parameters of the model. 
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