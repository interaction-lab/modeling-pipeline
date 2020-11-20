import neptune
import optuna
import joblib

# import torch
import neptunecontrib.monitoring.optuna as opt_utils
from neptunecontrib.api import log_table
from neptunecontrib.api import log_chart
import time

# from model_defs import TCNModel, RNNModel, LSTMModel, GRUModel
from data_utils import MyDataset, DataLoading
from model_training import ModelTraining
import matplotlib.pyplot as plt
import pandas as pd


def log_reports(metrics, columns):
    # Here is where we can get creative showing what we want
    # self.metrics = {"train": [], "val": [], "test": []}
    # self.columns

    # Create a figure showing metrics progress while training
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,15))
    df = pd.DataFrame(metrics["train"], columns=columns)
    columns_to_plot = [
        c for c in df.columns if ("support" not in c and "loss" not in c)
    ]
    df[columns_to_plot].plot(ax=axes[0])
    df["loss"].plot(ax=axes[0], secondary_y=True, color="black")
    axes[0].set_title("train")
    print(df)

    df = pd.DataFrame(metrics["val"], columns=columns)
    columns_to_plot = [
        c for c in df.columns if ("support" not in c and "loss" not in c)
    ]
    print(df)
    df[columns_to_plot].plot(ax=axes[1])
    df["loss"].plot(ax=axes[1], secondary_y=True, color="black")
    axes[1].set_title("val")
    # plt.show()
    experiment.log_image('diagrams', fig)
    # log_chart(name="performance", chart=fig)

    for k in ["train", "val", "test"]:
        df = pd.DataFrame(metrics[k], columns=columns)
        log_table(k, df)
        metrics_to_log = [
            "auROC",
            "AP",
            "support",
            "precision",
            "recall",
            "f1-score",
            "loss",
        ]
        for c in columns:
            for m in metrics_to_log:
                if m in c:
                    neptune.send_metric(f"{k}_{c}", df[c].iloc[-1])

def objective(trial):
    joblib.dump(study, folder_location)
    model_params = {}

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
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-4),
            "batch_size": trial.suggest_int("batch_size", 15, 25),
            "window": trial.suggest_int("window", 5, 30),
            "kern_size": trial.suggest_int("kern_size", 1, 5),
            "dropout": 0.25,
            "epochs": 2,
        }
    if model in ["rnn", "gru", "lstm"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 15, 25),
            "window": trial.suggest_int("window", 10, 30),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
            "dropout": 0.25,
            "epochs": 2,
        }

    try:
        start = time.time()
        dataset = MyDataset(df, window=model_params["window"], labels=params["classes"])
        # Hand tunable params:
        model_params["patience"] = 1
        model_params["weight_classes"] = True

        model_params["model"] = model
        model_params["target_names"] = params["classes"]
        model_params["num_features"] = dataset.df.shape[1]
        model_params["class_weights"] = dataset.weights
        model_params["num_classes"] = len(model_params["target_names"])

        trainer = ModelTraining(model_params, dataset, trial, verbose=True)
        auc_roc = trainer.train_and_eval_model()
        log_reports(trainer.metrics, trainer.columns)

        print(f"Round completed in {(time.time()-start)/60} min")
        return auc_roc
    except ValueError as E:
        print("Trial Failed!")
        print(E)
        return 0


# Configure Data Loader
config = "./config/data_loader_config.yml"
data_loader = DataLoading(config)
df = data_loader.get_all_sessions()

# Start up Neptune
neptune.init("cmbirmingham/Update-Turn-Taking")
neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)


# ***********PARAMETERS TO MESS WITH***********
EXP_NAME = "still_fixing"
num_trials = 15
classes = ["speaking"]
# All models ["rnn", "lstm", "gru", "mlp", "forest", "tree", "tcn", "knn", "xgb"]
models_to_try = ["tcn"]

# Record experiment details
params = {
    "classes": classes,
    "pruner": "no pruning",
    "trials": f"{num_trials}",
}
tags = [
    EXP_NAME,
    f"{len(data_loader.config['sessions'])} sessions",
    "unwindowed",
    "unnormalized",
    "comp: chris-personal",
]


for model in models_to_try:
    print(
        f"***********Creating study for {model} with {len(data_loader.config['sessions'])} sessions***********"
    )
    experiment = neptune.create_experiment(
        name=f"{model}_{EXP_NAME}",
        params=params,
        upload_source_files=[
            "sweep.py",
            "model_training.py",
            "model_defs.py",
            "data_utils.py",
            config,
        ],
    )
    folder_location = "./studies/study_{}_{}.pkl".format(model, EXP_NAME)
    for t in tags:
        neptune.append_tag(t)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=num_trials, callbacks=[neptune_callback])
    neptune.stop()
