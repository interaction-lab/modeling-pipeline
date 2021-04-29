# import neptune
import neptune.new as neptune
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import torch
from joblib import dump, load
import itertools

import neptunecontrib.monitoring.optuna as opt_utils
from optuna.samplers import TPESampler

from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.common.function_utils import timeit
from pipeline.modeling.model_training import ModelTraining
from .custom_dataset import MakeTurnsDataset as MTD
import argparse
from itertools import combinations

parser = argparse.ArgumentParser(description='Sweep hyperparams')
parser.add_argument('-m','--model', help='model to use, defaults to none', required=False, default=None)
parser.add_argument('-w','--window', help='Description for bar argument', required=False, default=None)
args = vars(parser.parse_args())

def log_reports(metrics, columns, log_to_neptune, verbose=False):
    # Here is where we can get creative showing what we want
    print("Logging reports")
    for k in ["train", "val", "test"]:
        if k == "test" or model in ["forest", "tree", "mlp", "knn", "xgb"]:
            df = pd.DataFrame([metrics[k]], columns=columns)
        else:
            df = pd.DataFrame(metrics[k], columns=columns)
        if log_to_neptune:
            # log_table(k, df)
            run['metrics/df'] = neptune.types.File.as_html(df)
        else:
            print(k, df)
        metrics_to_log = [
            "auROC",
            "AP",
            "support",
            "precision",
            "recall",
            "f1-score",
            "loss",
            "conf",
        ]
        for c in columns:
            for m in metrics_to_log:
                if m in c:
                    if log_to_neptune:
                        # neptune.send_metric(f"{k}_{c}", df[c].iloc[-1])
                        run[f"metrics/{k}/{c}"].log(df[c].iloc[-1])
                        # print(f"TO NEPTUNE AND BEYOND {k}_{c}", df[c].iloc[-1])
                    elif verbose:
                        print(f"{k}_{c}", df[c].iloc[-1])
    if model in ["forest", "tree", "mlp", "knn", "xgb"]:
        df_train = pd.DataFrame([metrics["train"]], columns=columns)
        df_val = pd.DataFrame([metrics["val"]], columns=columns)
    else:
        df_train = pd.DataFrame(metrics["train"], columns=columns)
        df_val = pd.DataFrame(metrics["val"], columns=columns)
    if df_val.shape[1] == 1:
        # Don't plot single point graphs
        return
    else:
        columns_to_plot = [
            c for c in df_train.columns if ("auROC" in c or "AP" in c or "f1" in c)
        ]
        # Create a figure showing metrics progress while training
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
        df_train[columns_to_plot].plot(ax=axes[0])
        df_train["loss"].plot(ax=axes[0], secondary_y=True, color="black")
        axes[0].set_title("train")

        df_val[columns_to_plot].plot(ax=axes[1])
        df_val["loss"].plot(ax=axes[1], secondary_y=True, color="black")
        axes[1].set_title("val")
        if verbose:
            plt.show()
        if log_to_neptune:
            run["metrics/diagrams"].log(fig)
    return

def set_model_params(model):
    model_params = {}
    if model in ["xgb"]:
        model_params = {
            "max_depth": 12,
            "booster":  "dart",
            "window": WINDOW,#trial.suggest_int("window", 1, MAX_HISTORY),
            "learning_rate": .001,
        }
    if model in ["rnn", "gru", "lstm", "tcn"]:
        model_params = {
            "num_layers": 6,
            "lr": 5e-5,
            "batch_size": 25,
            "window": WINDOW,#trial.suggest_int("window", 3, MAX_HISTORY),
            "kern_size": 5,
            "dropout": 0.25,
            "epochs": 200,
        }

    model_params["patience"] = PATIENCE
    model_params["weight_classes"] = WEIGHT_CLASSES
    model_params["model"] = model
    model_params["class_names"] = params["classes"]
    model_params["num_classes"] = len(model_params["class_names"])
    model_params["classes_org"] = LABELS_CLASSES

    return model_params

def search_model_params(trial, model):
    model_params = {}

    if model in ["tree", "forest"]:
        model_params = {
            # Tree/Forest Params
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_loguniform("min_samples", 1e-3, 0.5),
            "window": WINDOW,#trial.suggest_int("window", 1, MAX_HISTORY),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
    if model in ["knn"]:
        model_params = {
            # KNN params
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 16),
            "leaf_size": trial.suggest_int("leaf_size", 15, 45),
            "window": WINDOW,#trial.suggest_int("window", 1, MAX_HISTORY),
        }
    if model in ["xgb"]:
        model_params = {
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "window": WINDOW,#trial.suggest_int("window", 1, MAX_HISTORY),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.2),
        }
    if model in ["mlp"]:
        model_params = {
            # MLP Params
            "width": trial.suggest_int("width", 10, 100),
            "depth": trial.suggest_int("depth", 2, 7),
            "window": WINDOW,#trial.suggest_int("window", 1, MAX_HISTORY),
            "activation": trial.suggest_categorical(
                "activation", ["logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-1),
        }
    if model in ["tcn"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-4),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": WINDOW,#trial.suggest_int("window", 3, MAX_HISTORY),
            "kern_size": trial.suggest_int("kern_size", 1, 5),
            "dropout": 0.25,
            "epochs": 200,
        }
    if model in ["rnn", "gru", "lstm"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": WINDOW,#trial.suggest_int("window", 3, MAX_HISTORY),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
            "dropout": 0.25,
            "epochs": 200,
        }

    model_params["patience"] = PATIENCE
    model_params["weight_classes"] = WEIGHT_CLASSES
    model_params["model"] = model
    model_params["class_names"] = params["classes"]
    model_params["num_classes"] = len(model_params["class_names"])
    model_params["classes_org"] = LABELS_CLASSES

    return model_params

def read_dataset():
    train_df = pd.read_feather(FDF_PATH+".train")
    val_df = pd.read_feather(FDF_PATH+".val")
    test_df = pd.read_feather(FDF_PATH+".test")

    dataset = TimeSeriesDataset(
        train_df,
        val_df,
        test_df,
        labels=ALL_CLASSES,
        shuffle=SHUFFLE,
        subsample_perc=75,
    )
    return dataset

def save_model(model_name, model):
    if model_name in ["forest","tree","xgb"]:
        dump(model, 'model.pt') 
    if model in ["rnn","lstm","gru","tcn"]:
        torch.save(model.state_dict(), "model.pt")


@timeit
def opt_objective(trial):
    model_params = search_model_params(trial, model)
    dataset = read_dataset()
    dataset.setup_dataset(window=model_params["window"])

    model_params["subsample_perc"] = 75
    model_params["num_features"] = dataset.train_df.shape[1]
    model_params["class_weights"] = dataset.weights

    trainer = ModelTraining(model_params, dataset, trial, verbose=True)
    summary_metric = trainer.train_and_eval_model()
    save_model(model, trainer.model)

    log_reports(trainer.metrics, trainer.metrics_names, LOG_TO_NEPTUNE)

    if LOG_TO_NEPTUNE:
        run["metrics/dataset_shape"].log(dataset.train_df.shape[0])
        run["model_params"] = model_params

        opt_hist = optuna.visualization.plot_optimization_history(study)
        run["optuna/opt_history"].upload(opt_hist)
        par_coord = optuna.visualization.plot_parallel_coordinate(study)
        run["optuna/parallel_coord"].upload(par_coord)

        run[f'model_checkpoints/model-{summary_metric:.03f}.pt'].upload('model.pt')

    return summary_metric

@timeit
def cross_validate(k=9):
    num_sessions = 27
    fold_size = int(num_sessions/k)
    flatten = itertools.chain.from_iterable

    assert k>1, "must have at least two folds"
    for i in range(k):
        sets = [[f+n*fold_size+1 for f in range(fold_size)] for n in range(k)]

        val_sessions = sets.pop(i)
        train_sessions = list(flatten(sets))
        MTD(train_sessions,val_sessions, FDF_PATH, features=FEATURES)
        model_params = set_model_params(model)
        dataset = read_dataset()
        dataset.setup_dataset(window=model_params["window"])

        model_params["subsample_perc"] = 75
        model_params["num_features"] = dataset.train_df.shape[1]
        model_params["class_weights"] = dataset.weights

        trainer = ModelTraining(model_params, dataset, None, verbose=True)
        summary_metric = trainer.train_and_eval_model()
        save_model(model, trainer.model)

        log_reports(trainer.metrics, trainer.metrics_names, LOG_TO_NEPTUNE)

        if LOG_TO_NEPTUNE:
            run["metrics/dataset_shape"].log(dataset.train_df.shape[0])
            run["model_params"] = model_params

            run[f'model_checkpoints/{i}-model-{summary_metric:.03f}.pt'].upload('model.pt')

    return summary_metric


# ********************************************************************************
# *****************PARAMETERS TO CUSTOMIZE****************************************
# These parameters are used for controlling the sweep as
# well as describing the experiment for tracking in Neptune
# ********************************************************************************
# FDF_PATH = "./data/feathered_data/tmp-a.feather"
EXP_NAME = "cross-validate-asd"
COMPUTER = "laptop"

# Current models ["tree", "forest", "xgb", "gru", "rnn", "lstm", "tcn", "mlp"]
# Not working: "mlp", "knn"
available_models = [
    "tree",
    "forest",
    "xgb",
    "tcn",
    "rnn",
    "lstm",
    "gru",
]  

if args["model"]:
    assert args["model"] in available_models, "Model must be among list of available models"
    models_to_try = [args["model"]]
else:
    models_to_try = [
        "xgb",
        "gru",
        "tcn"
    ]
available_windows = [5,12,25]
if args["window"]:
    args["window"] = int(args["window"])
    assert args["window"] in available_windows, "window must be among list of available windows"
    WINDOWS = [args["window"]]
else:
    WINDOWS = available_windows


NUM_TRIALS = 5  # Number of trials to search for each model
PATIENCE = 2  # How many bad epochs to run before giving up

# Each class should be a binary column in the df
LABELS_CLASSES = {
    "speech": ["speaking"],
    # "turns": ["taking", "yielding", "holding", "listening"],
    # "uttertype": ["turn", "backchannel", "listening"]
}

# List of all classes
ALL_CLASSES = [i for _, v in LABELS_CLASSES.items() for i in v]

WEIGHT_CLASSES = True  # Weight loss against class imbalance
SHUFFLE = False
NORMALIZE = True  # Normalize entire dataset (- mean & / std dev)
OPTUNA_SEARCH = False
LOG_TO_NEPTUNE = True

# possible_features = ["at","ang","head","perfectmatch","syncnet"]
possible_features = ["at","ang","perfectmatch","syncnet"]
all_possible = []
for i in range(1,len(possible_features)+1):
    comb = combinations(possible_features,i)
    for i in list(comb): 
        all_possible.append(list(i))
# all_possible = [["syncnet"], ["ang","syncnet"],["perfectmatch"], ["ang","perfectmatch"]]
# all_possible = [["ang","syncnet"]]

for WINDOW in WINDOWS:
    # if WINDOW==25:
    #     all_features = all_possible[9:]
    # else:
    #     all_features = all_possible
    # all_features = all_possible
    for FEATURES in all_possible:
        # ***********************************************************************************
        # *****************Setup Experimental Details****************************************
        # Load the data here so it is not reloaded in each call to
        # optimize().
        # Set up experimental parameters to be shared with neptune.
        # Hyperparameters are set (and recorded) in optimize().
        # ***********************************************************************************
        window_config = f"examples/active_speaker/{EXP_NAME}_configs/windowing_example.yml"
        FDF_PATH = f"./data/feathered_data/tmp-{WINDOW}-{'-'.join(FEATURES)}.feather"

        # MTD has two responsibilities - to load the df and return a dataset
        MTD(range(1,20),range(20,28), FDF_PATH, features=FEATURES)


        # Record experimental details for Neptune
        params = {
            "trials": f"{NUM_TRIALS}",
            "pruner": "no pruning",  # See optuna.create_study
            "classes": ALL_CLASSES,
            "patience": PATIENCE,
            "weight classes": WEIGHT_CLASSES,
        }
        tags = [
            COMPUTER,run = neptune.init(project='cmbirmingham/cross-validate-asd')
            str(WINDOW),
        ]
        tags = tags + FEATURES


        # ***************************************************************************
        # *****************Run The Experiment****************************************
        # Here were try and optimize the hyperparams of each
        # model we are training
        # ***************************************************************************
        # Start up Neptune, init call takes the name of the sandbox
        # Neptune requires that you have set your api key in the terminal
        for model in models_to_try:
            tags.append(model)

            if LOG_TO_NEPTUNE:
                run = neptune.init(f"cmbirmingham/{EXP_NAME}", name="test-run", tags=tags)
                run["parameters"] = params
                for t in tags:
                    run["sys/tags"].add(t)

            if OPTUNA_SEARCH:
                print(f"***********Creating study for {model} ***********")
                study = optuna.create_study(
                    direction="maximize", 
                    pruner=optuna.pruners.NopPruner(), 
                    sampler=TPESampler(seed=10)
                )

                study.optimize(opt_objective, n_trials=NUM_TRIALS)

                if LOG_TO_NEPTUNE:
                    importance = optuna.visualization.plot_param_importances(study)
                    run["optuna/param_importance"].upload(importance)
                    run.stop()

                tags.remove(model)
            else:
                cross_validate()
                if LOG_TO_NEPTUNE:
                    run.stop()