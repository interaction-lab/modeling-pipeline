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
            "class_weights": class_weights,
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
            "class_weights": class_weights,
            "dropout": 0.25,
            "epochs": 200,  # trial.suggest_int("epochs", 5, 35),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
        }
    model_params["model"] = model
    model_params["num_features"] = 89
    model_params["patience"] = 15
    model_params["target_names"] = [
        # "silent",
        "talking",
        # "about_to_talk",
        "almost_finished",
    ]
    model_params["num_classes"] = len(model_params["target_names"])
    try:
        start = time.time()
        dataset = MyDataset(
            df,
            window=model_params["window"],
            pad_out=True,
            num_labels=model_params["num_classes"],
            continuous_labels=False,
            labels_at_end=True,
            augment=True,
        )
        print(f"dataset loaded with sessions: {params['sessions']}")
        print(f"Dataset loaded in {(time.time()-start)/60} min")
        trainer = ModelTraining(model_params, dataset, trial, verbose=True)
        auc_roc = trainer.train_and_eval_model()
        print(f"Round completed in {(time.time()-start)/60} min")
        return auc_roc
    except ValueError as E:
        print("Trial Failed!")
        print(E)
        return 0


####################
start = time.time()
dl = DataLoading()

neptune.init("cmbirmingham/turn-taking-sandbox2")


neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)
print(f"Neptune loaded in {(time.time()-start)/60} min")

####################
start = time.time()
# PARAMETERS TO MESS WITH
params = {"trials": 35, "sessions": (1, 6)}
class_weights = [1, 2]
df = dl.get_one_person_talking_dataset(sessions=params["sessions"], hard=True)
print(f"Dataframe loaded in {(time.time()-start)/60} min")


name = "gru_exploration"
trials = params["trials"]

# for model in ["rnn", "lstm", "gru", "mlp", "forest", "tree", "tcn", "knn"]:
# for model in ["xgb", "tcn", "mlp", "forest", "tree", "gru"]:
# for model in ["tcn", "xgb", "mlp", "forest", "tree", "gru"]:
for model in ["gru"]:
    ####################
    start = time.time()
    print(f"creating {model} study")
    neptune.create_experiment(
        name="{}_{}".format(model, name),
        params=params,
        upload_source_files=[
            "sweep.py",
            "model_training.py",
            "model_defs.py",
            "data_utils.py",
        ],
    )
    folder_location = "./studies/study_{}_{}.pkl".format(model, name)
    neptune.append_tag(model)
    neptune.append_tag("five sessions")
    neptune.append_tag("augmented")
    neptune.append_tag("concise")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    print(f"Study created in {(time.time()-start)/60} min")

    study.optimize(objective, n_trials=trials, callbacks=[neptune_callback])

    neptune.stop()
