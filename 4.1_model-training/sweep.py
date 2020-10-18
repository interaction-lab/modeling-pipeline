import neptune
import optuna
import joblib
import torch
import neptunecontrib.monitoring.optuna as opt_utils
import time

import data_utils
from data_utils import MyDataset, DataLoading

import model_defs
from model_defs import TCNModel, RNNModel, LSTMModel, GRUModel

import model_training
from model_training import ModelTraining


def objective(trial):
    joblib.dump(study, folder_location)

    if model in ["tree", "forest"]:
        model_params = {
            # Tree/Forest Params
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_loguniform("min_samples", 1e-3, 0.5),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
        model_params["window"] = 1
    if model in ["knn"]:
        model_params = {
            # KNN params
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 16),
            "leaf_size": trial.suggest_int("leaf_size", 15, 45),
        }
        model_params["window"] = 1
    if model in ["xgb"]:
        model_params = {
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        }
        model_params["window"] = 1
    if model in ["mlp"]:
        model_params = {
            # MLP Params
            "width": trial.suggest_int("width", 10, 100),
            "depth": trial.suggest_int("depth", 2, 5),
            "window": trial.suggest_int("window", 10, 30),
            "activation": trial.suggest_categorical(
                "activation", ["logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-1),
        }
        model_params["window"] = 1
    if model in ["tcn"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 6, 12),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": trial.suggest_int("window", 10, 30),
            "weight_classes": trial.suggest_categorical("weighting", [True, False]),
            "dropout": 0.25,
            "epochs": 200,  # trial.suggest_int("epochs", 5, 35),
            "kern_size": trial.suggest_int("kern_size", 1, 5),
        }
    if model in ["rnn", "gru", "lstm"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 15, 25),
            "window": trial.suggest_int("window", 10, 90),
            "weight_classes": trial.suggest_categorical("weighting", [True, False]),
            "dropout": 0.25,
            "epochs": 200,  # trial.suggest_int("epochs", 5, 35),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
        }

    model_params["model"] = model
    model_params["num_features"] = 201
    model_params["patience"] = 2
    model_params["target_names"] = params["classes"]
    model_params["num_classes"] = len(model_params["target_names"])

    try:
        start = time.time()
        dataset = MyDataset(df, window=model_params["window"], labels=params["classes"])
        model_params["num_features"] = dataset.df.shape[1]
        trainer = ModelTraining(model_params, dataset, trial, verbose=True)
        auc_roc = trainer.train_and_eval_model()

        print(f"Round completed in {(time.time()-start)/60} min")
        return auc_roc
    except ValueError as E:
        print("Trial Failed!")
        print(E)
        return 0


data_loader = DataLoading()
neptune.init("cmbirmingham/Update-Turn-Taking")
neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

####################
# All models ["rnn", "lstm", "gru", "mlp", "forest", "tree", "tcn", "knn", "xgb"]

# PARAMETERS TO MESS WITH
name = "first_attempt"
params = {"classes": ["speaking"]}
weight_classes = [1]
num_trials = 5
models_to_try = ["tcn"]


df = data_loader.get_all_sessions()

for model in models_to_try:
    ####################
    print(f"creating {model} study")
    neptune.create_experiment(
        name="{}_{}".format(model, name),
        params=params,
        upload_source_files=[
            "sweep.py",
            "model_training.py",
            "model_defs.py",
            "data_utils.py",
            "data_loader_config.yml",
        ],
    )
    folder_location = "./studies/study_{}_{}.pkl".format(model, name)
    neptune.append_tag(model)
    neptune.append_tag("all sessions")
    neptune.append_tag("Not Normalized")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())

    study.optimize(objective, n_trials=num_trials, callbacks=[neptune_callback])

    neptune.stop()
